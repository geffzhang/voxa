namespace Voxa.Audio.Diarization;

/// <summary>
/// Streaming (online) speaker diarization (VLS-011) — "who spoke when" as the audio arrives, instead of
/// re-diarizing a finished recording. It keeps three kinds of state and nothing else:
///
/// <list type="number">
/// <item><b>Incremental audio</b> — a fixed-capacity <see cref="AudioRingBuffer"/> addressed by absolute sample
/// index, plus the next window to run and the commit frontier. Audio is never accumulated for the whole session and
/// never re-processed.</item>
/// <item><b>Window overlap</b> — a window is trusted for the slice in its centre (the "owned" region: the part both it
/// and its neighbours saw), and a turn it reports outside that slice is published as provisional. That is where the
/// lookahead comes from: <c>(window − hop) / 2</c> seconds behind the newest audio.</item>
/// <item><b>Session speakers</b> — <see cref="SpeakerPrototypes"/>, matching each turn's embedding to a stable id,
/// plus the turns that are still in progress.</item>
/// </list>
///
/// <para><b>Why local turns and not frames.</b> The segmentation model's speaker channels are local to a window
/// (channel 0 in one window is not channel 0 in the next), so frames cannot be accumulated across windows by channel.
/// A run of frames on one local channel is a <i>local turn</i>: it covers a known audio span, that span is embedded
/// once, and the embedding is what carries identity across windows. Overlap survives because two local channels may
/// cover the same frames, and both turns are published.</para>
///
/// <para><b>A turn is published once and then extended.</b> The first window that reports a turn creates its segment
/// — provisional, labelled by continuity if it has not yet accumulated enough audio to be described — and later
/// windows extend the same segment (same id) until the turn's end lands in an owned region, at which point the
/// segment is final. A speaker merge is the other revision source, bounded by the revision window. Consumers key on
/// <see cref="SpeakerSegmentUpdate.Id"/> and treat a higher revision as a replacement.</para>
///
/// <para>Not thread-safe: one instance per session stream, driven by a single consumer (the frame processor's worker
/// task). It holds no model runtime — the segmenter and embedder are seams.</para>
/// </summary>
public sealed class StreamingDiarizer
{
    private readonly ISpeakerMaskSegmentation _segmentation;
    private readonly ISpeakerEmbedding _embedding;
    private readonly StreamingDiarizerConfig _config;

    private readonly int _windowSamples;
    private readonly int _hopSamples;
    private readonly int _frameShift;
    private readonly int _frameOffset;
    private readonly int _ownedOffsetSamples; // distance from a window's start to the region it decides

    private readonly AudioRingBuffer _audio;
    private readonly SpeakerPrototypes _prototypes;
    private readonly float[] _windowScratch;
    private float[] _embeddingScratch = []; // reused: a decision must not allocate a turn's worth of audio

    private readonly Dictionary<string, Segment> _history = []; // published segments, in creation order
    private readonly Dictionary<int, string> _channelMap = [];  // local channel -> segment it continued last window
    private readonly List<Segment> _open = [];                  // turns still in progress
    private readonly HashSet<string> _touched = [];             // segments this window reported

    private readonly List<SpeakerSegmentUpdate> _updates = [];
    private readonly List<SpeakerMergeInfo> _merges = [];

    private long _nextWindowStart;
    private long _frontier;
    private int _segmentCounter;

    public StreamingDiarizer(
        ISpeakerMaskSegmentation segmentation,
        ISpeakerEmbedding embedding,
        StreamingDiarizerConfig? config = null)
    {
        _segmentation = segmentation ?? throw new ArgumentNullException(nameof(segmentation));
        _embedding = embedding ?? throw new ArgumentNullException(nameof(embedding));
        _config = config ?? new StreamingDiarizerConfig();

        if (_config.SampleRate != segmentation.SampleRate)
            throw new ArgumentException(
                $"Diarizer sample rate {_config.SampleRate} Hz does not match the segmentation engine's " +
                $"{segmentation.SampleRate} Hz. Resample upstream.", nameof(config));
        if (embedding.EmbeddingDim <= 0)
            throw new ArgumentException("The embedding engine must declare a positive EmbeddingDim.", nameof(embedding));

        _windowSamples = _config.ResolveWindowSamples(segmentation);
        _hopSamples = _config.ResolveHopSamples(_windowSamples);
        _frameShift = segmentation.FrameShiftSamples;
        _frameOffset = segmentation.FrameOffsetSamples;

        ValidateGeometry();

        // The owned region is the window's centre: (window - hop) / 2 in, one hop wide.
        _ownedOffsetSamples = (_windowSamples - _hopSamples) / 2;

        int capacity = Math.Max(_windowSamples + _hopSamples, (int)(_config.MaxBufferedSeconds * _config.SampleRate));
        _audio = new AudioRingBuffer(capacity);
        _windowScratch = new float[_windowSamples];
        _prototypes = new SpeakerPrototypes(_config);
    }

    /// <summary>Sample rate of the pushed audio (equal to the segmentation engine's).</summary>
    public int SampleRate => _config.SampleRate;

    /// <summary>Model window length in samples, after resolving the config against the engine's geometry.</summary>
    public int WindowSamples => _windowSamples;

    /// <summary>Advance between windows, in samples.</summary>
    public int HopSamples => _hopSamples;

    /// <summary>
    /// Absolute seconds up to which every segment has been decided. Audio newer than this has at most a
    /// <i>provisional</i> label; the distance to <see cref="SamplesSeen"/> is the lookahead this capability pays.
    /// </summary>
    public double CommitFrontierSeconds => _frontier / (double)SampleRate;

    /// <summary>Absolute samples ingested so far, including those reported through <see cref="NotifyGap"/>.</summary>
    public long SamplesSeen => _audio.End;

    /// <summary>Samples currently retained by the bounded audio buffer.</summary>
    public long BufferedSamples => _audio.Count;

    /// <summary>Speaker ids alive in this session, in creation order (earlier ids survive a merge).</summary>
    public IReadOnlyList<int> SpeakerIds => _prototypes.SpeakerIds;

    /// <summary>
    /// Feed audio at <see cref="SampleRate"/>. Returns everything the windows this push completed published; most
    /// pushes publish nothing (the window is still filling).
    /// </summary>
    public DiarizerUpdate Push(ReadOnlySpan<float> audio)
    {
        _updates.Clear();
        _merges.Clear();

        if (audio.Length > 0) _audio.Write(audio);

        while (SamplesSeen >= _nextWindowStart + _windowSamples)
        {
            RunWindow(_nextWindowStart, flush: false);
            _nextWindowStart += _hopSamples;
            RetainAudio();
        }

        RetainAudio();

        return Result();
    }

    // Two readers need the buffer: the window schedule, which starts at _nextWindowStart, and the closing embedding
    // of a turn, which reaches back to that turn's start. Audio is dropped only when neither can use it, so a turn
    // longer than a window is still described by the whole turn and not by whatever tail happened to survive.
    private void RetainAudio()
    {
        long horizon = SamplesSeen - (long)(_config.MaxBufferedSeconds * SampleRate);
        _audio.DiscardBefore(Math.Min(_nextWindowStart, horizon));
    }

    /// <summary>
    /// Report that <paramref name="droppedSamples"/> of audio were lost upstream (the caller's bounded queue
    /// overflowed). The timeline still advances — timestamps keep meaning what they meant — but the window schedule
    /// restarts after the hole and every open turn is closed, because whatever continued in the lost audio cannot be
    /// recovered. A gap is a discontinuity, never silently stitched.
    /// </summary>
    public DiarizerUpdate NotifyGap(int droppedSamples)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(droppedSamples);
        _updates.Clear();
        _merges.Clear();
        if (droppedSamples == 0) return Result();

        foreach (var segment in _open.ToArray()) Finalize(segment);
        _open.Clear();

        // Advance the timeline with silence: bounded writes, so a long gap cannot allocate.
        var silence = new float[Math.Min(droppedSamples, Math.Max(1, SampleRate))];
        int remaining = droppedSamples;
        while (remaining > 0)
        {
            int chunk = Math.Min(remaining, silence.Length);
            _audio.Write(silence.AsSpan(0, chunk));
            remaining -= chunk;
        }

        _nextWindowStart = SamplesSeen;
        _frontier = SamplesSeen;
        _audio.DiscardBefore(SamplesSeen);
        return Result();
    }

    /// <summary>
    /// End of stream: run the remaining windows (a zero-padded tail, like the batch decoder's last chunk), publish the
    /// tail, and commit every open turn. Safe to call once; a later <see cref="Push"/> starts a fresh schedule.
    /// </summary>
    public DiarizerUpdate Flush()
    {
        _updates.Clear();
        _merges.Clear();

        // Each iteration owns exactly the next hop, so the schedule continues where the stream left it: windows never
        // step back over audio that was already run. The window itself reaches past the end of the audio, which the
        // ring buffer zero-fills — that is the padded tail, not silence this engine inferred.
        while (_frontier < SamplesSeen)
        {
            long start = _frontier - _ownedOffsetSamples;
            if (start < _audio.Oldest) start = _audio.Oldest;
            if (start < 0) start = 0;

            long previous = _frontier;
            RunWindow(start, flush: true);
            if (_frontier <= previous) _frontier = SamplesSeen; // belt and braces: the loop must terminate
        }

        foreach (var segment in _open.ToArray()) Finalize(segment);
        _open.Clear();

        _nextWindowStart = Math.Max(_nextWindowStart, SamplesSeen);
        return Result();
    }

    // ---------------------------------------------------------------- window processing

    private void RunWindow(long windowStart, bool flush)
    {
        _audio.CopyTo(windowStart, _windowScratch);
        var masks = _segmentation.Analyze(_windowScratch);
        if (masks.FrameCount == 0) return;

        long ownedStart = windowStart + _ownedOffsetSamples;
        long ownedEnd = Math.Min(ownedStart + _hopSamples, SamplesSeen);
        if (ownedEnd <= ownedStart && !flush) return;

        _touched.Clear();
        long firstFrame = windowStart / _frameShift;
        var channelMap = new Dictionary<int, string>(_channelMap);

        foreach (var turn in ExtractTurns(masks))
        {
            long turnStartFrame = firstFrame + turn.StartFrame;
            long turnEndFrame = firstFrame + turn.EndFrame;
            long turnStart = turnStartFrame * _frameShift + _frameOffset;
            long turnEnd = Math.Min(turnEndFrame * _frameShift + _frameOffset, SamplesSeen);
            if (turnEnd <= turnStart) continue;

            // The turn ends at or before the frontier, so the audio that decides it is already behind us: nothing
            // this window says can change it. Flush windows overlap the schedule the stream already ran, and this
            // is what stops them re-reporting finished turns as new ones.
            if (turnEnd <= _frontier) continue;

            var segment = Associate(turn, firstFrame, turnStartFrame, turnEnd, out bool alreadyCommitted);
            if (alreadyCommitted) continue;

            if (segment is null)
            {
                segment = CreateSegment(turnStart, turnEnd, turnStartFrame, turnEndFrame);
            }
            else
            {
                segment.EndSample = Math.Max(segment.EndSample, turnEnd);
                segment.LastFrame = Math.Max(segment.LastFrame, turnEndFrame);
            }

            // How clearly this turn's own track is present, as the strongest window that saw it: a window that only
            // catches the turn's closing frames (or its tail in silence) would understate a voice that was obvious
            // a moment ago, and confidence gates speaker creation.
            segment.Activity = Math.Max(segment.Activity, turn.Confidence);

            channelMap[turn.Speaker] = segment.Id;
            _touched.Add(segment.Id);

            // The turn's end landed inside the region this window decides: this is the last word on its geometry.
            bool ending = flush || turnEnd <= ownedEnd;

            // Describe the turn as soon as there is enough of it to describe — that early label is the whole point
            // of streaming — and once more when it ends, from the complete turn, which is the most accurate read
            // there will ever be. Only the closing read enrols the voice: a provisional match must not drag a
            // prototype toward a fragment, or an overlap would teach a settled speaker somebody else's voice.
            if (ShouldDecide(segment, ending)) DecideSpeaker(segment, enrol: ending);

            if (ending) Finalize(segment);
            else Publish(segment);
        }

        // A turn nothing reported for a full hop is over, even if its closing frames were lost to a window boundary.
        foreach (var segment in _open.ToArray())
        {
            if (segment.EndSample <= _frontier && !_touched.Contains(segment.Id)) Finalize(segment);
        }

        _channelMap.Clear();
        foreach (var (channel, id) in channelMap) _channelMap[channel] = id;

        _frontier = Math.Max(_frontier, flush ? Math.Min(ownedEnd, SamplesSeen) : ownedEnd);

        if (_prototypes.Merge(_frontier / (double)SampleRate) is { Count: > 0 } merges) ApplyMerges(merges);

        PruneHistory();
    }

    private Segment CreateSegment(long startSample, long endSample, long firstFrame, long lastFrame)
    {
        var (previousSpeaker, gap) = FindPrevious(startSample);
        var segment = new Segment
        {
            Id = $"seg-{++_segmentCounter}",
            StartSample = startSample,
            EndSample = endSample,
            FirstFrame = firstFrame,
            LastFrame = lastFrame,
            PreviousSpeaker = previousSpeaker,
            PreviousGap = gap,
            // A turn with no evidence yet borrows the speaker who was just talking — a backchannel, or a continuation
            // cut by a window boundary. It stays provisional (and this label is what consumers see) until the turn is
            // long enough to describe.
            Speaker = previousSpeaker >= 0 && gap <= _config.ContinuityWindow
                ? previousSpeaker
                : SpeakerSegmentUpdate.UnknownSpeaker,
        };
        segment.Confidence = segment.Speaker >= 0 ? 0.25 : 0;
        _history[segment.Id] = segment;
        _open.Add(segment);
        Publish(segment);
        return segment;
    }

    // Enough audio to describe a voice, judged against both the embedding floor and the speaker-creation evidence
    // floor: deciding earlier would mean deciding from a fragment and never revisiting it.
    private bool HasEvidence(Segment segment)
    {
        double required = Math.Max(_config.MinEmbeddingDuration, _config.MinNewSpeakerEvidence);
        return segment.EndSample - segment.StartSample >= GapSamples(required);
    }

    // A turn is described once when it first carries enough audio (provisional), and once when it ends. Nothing is
    // re-described mid-turn: an embedding costs a model call, and a label taken from a growing fragment churns.
    private bool ShouldDecide(Segment segment, bool ending)
    {
        if (ending) return !segment.Decided || segment.DecidedSample != segment.EndSample;
        return !segment.Decided && HasEvidence(segment);
    }

    private void DecideSpeaker(Segment segment, bool enrol)
    {
        double duration = (segment.EndSample - segment.StartSample) / (double)SampleRate;
        if (duration >= _config.MinEmbeddingDuration && EmbedSpan(segment.StartSample, segment.EndSample) is { } embedding)
        {
            var decision = _prototypes.Decide(
                embedding, duration, segment.Activity, segment.PreviousSpeaker, segment.PreviousGap, enrol);
            segment.Speaker = decision.Speaker;
            segment.Confidence = decision.LabelConfidence;
        }
        // Too short to describe a voice: keep the continuity label the segment was created with.
        segment.Decided = true;
        segment.DecidedSample = segment.EndSample;
    }

    private void Finalize(Segment segment)
    {
        segment.Extendable = false;
        _open.Remove(segment);
        Publish(segment);
    }

    // Every publication is a new revision of the same id: 0 for the first, then 1, 2, ... Consumers upsert by id, so
    // the numbering is what lets a late or duplicated delivery be recognised as stale.
    private void Publish(Segment segment)
    {
        _updates.Add(new SpeakerSegmentUpdate(
            segment.Id,
            segment.Speaker,
            Time(segment.StartSample),
            Time(segment.EndSample),
            IsFinal: !segment.Extendable,
            Revision: segment.Revision++,
            segment.Confidence));
    }

    private void ApplyMerges(IReadOnlyList<SpeakerMergeInfo> merges)
    {
        foreach (var merge in merges)
        {
            _merges.Add(merge);

            foreach (var segment in _history.Values)
            {
                if (segment.Speaker != merge.RetiredSpeaker) continue;
                segment.Speaker = merge.SurvivingSpeaker;
                Publish(segment); // the revision window only — PruneHistory keeps the list bounded
            }
        }
    }

    // Bound the segment history: everything older than the revision window may go, except turns still in progress
    // (a merge can never rewrite what is no longer retained — that is what the merge notice is for).
    private void PruneHistory()
    {
        const int minRetained = 128;
        if (_history.Count <= minRetained) return;

        double horizon = _frontier / (double)SampleRate - Math.Max(_config.RevisionWindow, 0);
        foreach (var id in _history.Keys.ToArray())
        {
            if (_history.Count <= minRetained) break;
            var segment = _history[id];
            if (segment.Extendable || Time(segment.EndSample) >= horizon) continue;
            _history.Remove(id);
        }
    }

    // ---------------------------------------------------------------- helpers

    // Which in-progress segment this run continues, or null when nothing can claim it.
    //
    // A run that begins at the window's left edge was already talking when the window opened: the model truncated
    // it, so the run's start says nothing about where the turn began and only the per-channel map can say what it
    // continues (with overlapping speech both channels cover the same frames, so frames alone cannot tell which
    // turn continues which segment). A run whose start sits inside the window is a sighting in its own right,
    // identified by its absolute start frame — that frame stays put as later windows re-report the same turn.
    private Segment? Associate(LocalTurn turn, long firstFrame, long turnStartFrame, long turnEnd, out bool alreadyCommitted)
    {
        alreadyCommitted = false;
        if (turn.StartFrame > 0) return FindOpen(turnStartFrame);

        if (!_channelMap.TryGetValue(turn.Speaker, out var id) || !_history.TryGetValue(id, out var previous))
            return null; // nothing was talking on this channel: the turn really does start here

        if (previous.Extendable && previous.LastFrame >= firstFrame - 1) return previous;

        // The channel's segment is already committed and this run ends where it ended: a later window is
        // re-reporting that finished turn (flush windows overlap their predecessors), not finding a second one.
        alreadyCommitted = !previous.Extendable && turnEnd <= previous.EndSample;
        return null;
    }

    // The same turn re-reported by a later window: the model reports a run from its own start, so the start frame
    // is the turn's identity while it stays inside the window. Two speakers overlapping start at different frames
    // (and if they ever share one, the first has already claimed its segment this window), so position alone
    // cannot fuse them.
    private Segment? FindOpen(long startFrame)
    {
        foreach (var segment in _open)
        {
            if (_touched.Contains(segment.Id)) continue;
            if (Math.Abs(segment.FirstFrame - startFrame) <= 1) return segment;
        }
        return null;
    }

    // The most recent published segment that ends at or before `beforeSample` — the speaker a short turn or a
    // backchannel is most likely continuing.
    private (int Speaker, double Gap) FindPrevious(long beforeSample)
    {
        int speaker = SpeakerSegmentUpdate.UnknownSpeaker;
        long best = long.MinValue;
        foreach (var segment in _history.Values)
        {
            if (segment.EndSample > beforeSample || segment.EndSample < best) continue;
            best = segment.EndSample;
            speaker = segment.Speaker;
        }
        double gap = speaker < 0 ? double.MaxValue : (beforeSample - best) / (double)SampleRate;
        return (speaker, gap);
    }

    // The turn's audio, read back from the buffer. A turn longer than the retained horizon is described by what is
    // still there — bounded memory means an unbounded turn cannot be embedded whole, and reporting the truncation
    // would only invite callers to act on a number they cannot control.
    private float[]? EmbedSpan(long startSample, long endSample)
    {
        long from = Math.Max(startSample, _audio.Oldest);
        int length = (int)Math.Min(endSample - from, int.MaxValue);
        if (length <= 0) return null;

        if (_embeddingScratch.Length < length) _embeddingScratch = new float[length];
        int copied = _audio.CopyTo(from, _embeddingScratch);
        return copied == 0 ? null : _embedding.Embed(_embeddingScratch.AsSpan(0, copied), SampleRate);
    }

    private double Time(long sample) => sample / (double)SampleRate;

    private long GapSamples(double seconds) => (long)(seconds * SampleRate);

    private DiarizerUpdate Result()
        => _updates.Count == 0 && _merges.Count == 0
            ? DiarizerUpdate.Empty
            : new DiarizerUpdate(_updates.ToArray(), _merges.ToArray());

    /// <summary>One run of frames on one local speaker channel inside a window.</summary>
    private readonly record struct LocalTurn(int Speaker, int StartFrame, int EndFrame, double Confidence);

    // Onset/offset hysteresis per local channel: open at >= Onset, close after Offset has held for
    // MinSilenceDuration. Shorter dips are absorbed (breaths, plosives), which is what keeps a speaker change from
    // being cut into two turns a few frames apart.
    private List<LocalTurn> ExtractTurns(SpeakerMaskWindow masks)
    {
        var turns = new List<LocalTurn>();
        int frames = masks.FrameCount;
        int speakers = _segmentation.MaxLocalSpeakers;
        int closeFrames = Math.Max(1, (int)Math.Round(_config.MinSilenceDuration * SampleRate / _frameShift));

        for (int s = 0; s < speakers; s++)
        {
            bool active = false;
            int startFrame = 0, silent = 0;
            double sum = 0;
            int counted = 0;

            for (int f = 0; f < frames; f++)
            {
                var row = masks.FrameProbabilities[f];
                float probability = s < row.Length ? row[s] : 0f;

                if (!active)
                {
                    if (probability >= _config.Onset)
                    {
                        active = true;
                        startFrame = f;
                        silent = 0;
                        sum = probability;
                        counted = 1;
                    }
                    continue;
                }

                sum += probability;
                counted++;
                if (probability >= _config.Offset)
                {
                    silent = 0;
                }
                else if (++silent >= closeFrames)
                {
                    turns.Add(new LocalTurn(s, startFrame, f - silent + 1, sum / counted));
                    active = false;
                }
            }

            if (active) turns.Add(new LocalTurn(s, startFrame, frames, sum / counted));
        }

        turns.Sort((x, y) => x.StartFrame.CompareTo(y.StartFrame));
        return turns;
    }

    private void ValidateGeometry()
    {
        if (_windowSamples <= 0 || _hopSamples <= 0 || _frameShift <= 0)
            throw new ArgumentException(
                $"Invalid diarization geometry: window={_windowSamples}, hop={_hopSamples}, frameShift={_frameShift}.");

        if (_hopSamples > _windowSamples / 3)
            throw new ArgumentException(
                $"Hop {_hopSamples} must be at most a third of the window {_windowSamples}: a window has to see the " +
                "whole region it decides.");

        if (_windowSamples % _frameShift != 0 || _hopSamples % _frameShift != 0
            || (_windowSamples - _hopSamples) / 2 % _frameShift != 0)
            throw new ArgumentException(
                $"Window {_windowSamples} / hop {_hopSamples} must both be multiples of the frame shift {_frameShift}, " +
                "and (window - hop) / 2 must land on the frame grid.");
    }

    /// <summary>Mutable state for one published segment (upserted by <see cref="Segment.Id"/>).</summary>
    private sealed class Segment
    {
        public required string Id { get; init; }
        public int Speaker { get; set; }
        public long StartSample { get; init; }
        public long EndSample { get; set; }
        public long FirstFrame { get; init; }
        public long LastFrame { get; set; }
        public int PreviousSpeaker { get; init; }
        public double PreviousGap { get; init; }
        public double Activity { get; set; } // strongest per-window activity this turn's track reached
        public int Revision { get; set; }
        public bool Extendable { get; set; } = true;
        public bool Decided { get; set; }
        public long DecidedSample { get; set; } // the end the last decision was taken from
        public double Confidence { get; set; }
    }
}
