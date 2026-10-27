using Voxa.Audio.Diarization;

namespace Voxa.Audio.Diarization.Tests;

/// <summary>
/// Streaming-diarization behaviour on a scripted two-voice conversation (VLS-011): stable ids across windows,
/// overlapping speech preserved, incremental state that does not depend on how the audio was chopped, a bounded
/// buffer, and the provisional→final revision protocol. No model, no network — the segmentation and embedding seams
/// are the tone fixtures.
/// </summary>
public class StreamingDiarizerTests
{
    private const int SampleRate = ToneSignal.SampleRate;
    private const int WindowSamples = 3_200; // 200 ms
    private const int HopSamples = 640;      // 40 ms ⇒ 80 ms lookahead

    private static StreamingDiarizerConfig Config(params (string Key, object Value)[] overrides)
    {
        var config = new StreamingDiarizerConfig
        {
            SampleRate = SampleRate,
            WindowSamples = WindowSamples,
            HopSamples = HopSamples,
            Onset = 0.5,
            Offset = 0.35,
            MinSilenceDuration = 0.04,
            MinSpeechDuration = 0.06,
            MinEmbeddingDuration = 0.08,
            MatchThreshold = 0.45,
            NewSpeakerThreshold = 0.62,
            MinNewSpeakerEvidence = 0.15,
            MinTurnConfidence = 0.5,
            ContinuityWindow = 0.3,
            MaxMergeGap = 0.1,
            MaxSpeakers = 10,
            RevisionWindow = 5,
            MergeThreshold = 0.3,
            MergeEvidence = 3,
            MaxBufferedSeconds = 8,
        };

        foreach (var (key, value) in overrides)
        {
            config = key switch
            {
                nameof(StreamingDiarizerConfig.MinEmbeddingDuration) => config with { MinEmbeddingDuration = (double)value },
                nameof(StreamingDiarizerConfig.MaxSpeakers) => config with { MaxSpeakers = (int)value },
                nameof(StreamingDiarizerConfig.MaxBufferedSeconds) => config with { MaxBufferedSeconds = (double)value },
                nameof(StreamingDiarizerConfig.ContinuityWindow) => config with { ContinuityWindow = (double)value },
                nameof(StreamingDiarizerConfig.MinNewSpeakerEvidence) => config with { MinNewSpeakerEvidence = (double)value },
                nameof(StreamingDiarizerConfig.Onset) => config with { Onset = (double)value },
                _ => throw new ArgumentOutOfRangeException(nameof(overrides), key, "Unsupported override."),
            };
        }

        return config;
    }

    private static (StreamingDiarizer Diarizer, ToneEmbedding Embedding) Create(StreamingDiarizerConfig? config = null)
    {
        var embedding = new ToneEmbedding();
        var diarizer = new StreamingDiarizer(new ToneMaskSegmentation(WindowSamples), embedding, config ?? Config());
        return (diarizer, embedding);
    }

    // Push in 20 ms chunks, the way the processor feeds it, folding every update into the consumer's state.
    private static SegmentTable Run(StreamingDiarizer diarizer, float[] audio, int chunkSamples = 320)
    {
        var table = new SegmentTable();
        for (int offset = 0; offset < audio.Length; offset += chunkSamples)
            table.Apply(diarizer.Push(audio.AsSpan(offset, Math.Min(chunkSamples, audio.Length - offset))));
        table.Apply(diarizer.Flush());
        return table;
    }

    [Fact]
    public void Alternating_speakers_keep_stable_ids_across_windows()
    {
        var (diarizer, _) = Create();
        var audio = ToneSignal.Build(3.2,
            (0.0, 1.0, 0.5, 0.0),   // speaker A
            (1.2, 2.2, 0.0, 0.5),   // speaker B
            (2.4, 3.2, 0.5, 0.0));  // speaker A again

        var table = Run(diarizer, audio);
        var finals = table.Final;

        Assert.Equal(3, finals.Count);
        Assert.All(finals, s => Assert.True(s.Confidence > 0.5, "each turn is long enough to be described"));

        Assert.Equal(0, finals[0].Speaker);
        Assert.Equal(1, finals[1].Speaker);
        Assert.Equal(0, finals[2].Speaker); // the id is the voice, not the turn
        Assert.Equal(new[] { 0, 1 }, diarizer.SpeakerIds);

        // Boundaries land on the script within a hop of lookahead.
        Assert.Equal(0.0, finals[0].Start, 1);
        Assert.Equal(1.0, finals[0].End, 1);
        Assert.Equal(1.2, finals[1].Start, 1);
        Assert.Equal(2.2, finals[1].End, 1);
        Assert.Equal(2.4, finals[2].Start, 1);
    }

    [Fact]
    public void Overlapping_speech_keeps_both_segments()
    {
        // Onset below the 0.5 that two equal voices each get in a full overlap: both channels must stay open, which is
        // the point of the test (the default 0.5 would make the overlap a knife-edge).
        var (diarizer, _) = Create(Config((nameof(StreamingDiarizerConfig.Onset), 0.4)));
        var audio = ToneSignal.Build(2.2,
            (0.0, 1.5, 0.5, 0.0),   // A alone, then A + B
            (1.0, 2.0, 0.0, 0.5));  // B starts while A is still talking

        var table = Run(diarizer, audio);
        var finals = table.Final;

        Assert.Equal(2, finals.Count);
        Assert.Equal(new[] { 0, 1 }, finals.Select(s => s.Speaker).OrderBy(x => x).ToArray());

        // The two turns genuinely overlap: a speech/silence collapse would have produced one merged region.
        double overlapStart = Math.Max(finals[0].Start, finals[1].Start);
        double overlapEnd = Math.Min(finals[0].End, finals[1].End);
        Assert.True(overlapEnd - overlapStart >= 0.4, $"expected a sustained overlap, got [{overlapStart}, {overlapEnd}]");
    }

    [Fact]
    public void Chunking_does_not_change_the_result()
    {
        var audio = ToneSignal.Build(2.6,
            (0.0, 0.9, 0.5, 0.0),
            (1.1, 1.9, 0.0, 0.5),
            (2.1, 2.6, 0.5, 0.0));

        var (chunked, _) = Create();
        var (whole, _) = Create();

        var fromChunks = Run(chunked, audio, chunkSamples: 320);            // 20 ms
        var fromWhole = Run(whole, audio, chunkSamples: audio.Length);      // one push

        Assert.Equal(Describe(fromChunks.Final), Describe(fromWhole.Final));
    }

    [Fact]
    public void Audio_buffer_stays_bounded_on_a_long_stream()
    {
        var (diarizer, _) = Create(Config((nameof(StreamingDiarizerConfig.MaxBufferedSeconds), 2.0)));
        var audio = ToneSignal.Build(12.0, (0.0, 12.0, 0.5, 0.0));

        for (int offset = 0; offset < audio.Length; offset += 1_600)
            diarizer.Push(audio.AsSpan(offset, Math.Min(1_600, audio.Length - offset)));

        // The session ran 12 s through a 2 s buffer: retained audio is bounded, the timeline is not.
        Assert.True(diarizer.BufferedSamples <= 2 * SampleRate, $"buffer grew to {diarizer.BufferedSamples} samples");
        Assert.Equal(audio.Length, diarizer.SamplesSeen);
    }

    [Fact]
    public void Flush_commits_the_tail_that_would_still_be_provisional()
    {
        var (diarizer, _) = Create();
        var audio = ToneSignal.Build(1.0, (0.0, 1.0, 0.5, 0.0));

        var table = new SegmentTable();
        for (int offset = 0; offset < audio.Length; offset += 320)
            table.Apply(diarizer.Push(audio.AsSpan(offset, Math.Min(320, audio.Length - offset))));

        var beforeFlush = table.All;
        Assert.Contains(beforeFlush, s => !s.IsFinal); // the tail is not decided yet

        table.Apply(diarizer.Flush());
        Assert.All(table.All, s => Assert.True(s.IsFinal, "flush must commit every open turn"));
    }

    [Fact]
    public void A_reported_gap_keeps_the_absolute_timeline()
    {
        var (diarizer, _) = Create();
        var first = ToneSignal.Build(0.8, (0.0, 0.8, 0.5, 0.0));
        var second = ToneSignal.Build(0.8, (0.0, 0.8, 0.0, 0.5));

        var table = new SegmentTable();
        foreach (var sample in first.Chunk(320)) table.Apply(diarizer.Push(sample));
        table.Apply(diarizer.NotifyGap(SampleRate)); // one second of audio lost upstream
        foreach (var sample in second.Chunk(320)) table.Apply(diarizer.Push(sample));
        table.Apply(diarizer.Flush());

        Assert.Equal(first.Length + SampleRate + second.Length, diarizer.SamplesSeen);

        var secondTurn = table.Final.Single(s => s.Speaker == 1);
        // The gap is timeline, not silence in the buffer: the second turn starts after it, not at zero.
        Assert.InRange(secondTurn.Start, 1.75, 1.95);
    }

    [Fact]
    public void A_short_turn_borrows_the_previous_speaker_instead_of_inventing_one()
    {
        var (diarizer, _) = Create(Config((nameof(StreamingDiarizerConfig.MinEmbeddingDuration), 0.2)));
        var audio = ToneSignal.Build(2.1,
            (0.0, 1.5, 0.5, 0.0),   // A
            (1.7, 1.85, 0.0, 0.5)); // 150 ms of B — long enough to be speech, too short to describe a voice

        var table = Run(diarizer, audio);

        Assert.Equal(new[] { 0 }, diarizer.SpeakerIds); // no new speaker from 150 ms
        var blip = table.Final.Last();
        Assert.Equal(0, blip.Speaker);
        Assert.True(blip.Confidence <= 0.3, "a borrowed label must not claim confidence");
    }

    [Fact]
    public void A_thin_turn_with_no_history_is_unknown()
    {
        var (diarizer, _) = Create(Config((nameof(StreamingDiarizerConfig.MinEmbeddingDuration), 0.2)));
        var audio = ToneSignal.Build(0.9, (0.1, 0.25, 0.0, 0.5)); // a 150 ms blip, nothing before it

        var table = Run(diarizer, audio);

        Assert.Empty(diarizer.SpeakerIds);
        var segment = Assert.Single(table.Final);
        Assert.Equal(SpeakerSegmentUpdate.UnknownSpeaker, segment.Speaker);
        Assert.Equal(0, segment.Confidence);
    }

    [Fact]
    public void The_speaker_cap_attaches_instead_of_creating()
    {
        var (diarizer, _) = Create(Config(
            (nameof(StreamingDiarizerConfig.MaxSpeakers), 1),
            (nameof(StreamingDiarizerConfig.MinNewSpeakerEvidence), 0.15)));

        var audio = ToneSignal.Build(2.2, (0.0, 1.0, 0.5, 0.0), (1.2, 2.2, 0.0, 0.5));
        var table = Run(diarizer, audio);

        Assert.Single(diarizer.SpeakerIds);
        Assert.All(table.Final, s => Assert.Equal(0, s.Speaker));
    }

    [Fact]
    public void The_same_audio_produces_the_same_updates()
    {
        var audio = ToneSignal.Build(2.0, (0.0, 1.0, 0.5, 0.0), (1.2, 2.0, 0.0, 0.5));

        var first = Run(Create().Diarizer, audio);
        var second = Run(Create().Diarizer, audio);

        Assert.Equal(Describe(first.Final), Describe(second.Final));
    }

    [Fact]
    public void Revisions_are_monotonic_per_segment_and_final_comes_last()
    {
        var (diarizer, _) = Create();
        var audio = ToneSignal.Build(1.6, (0.0, 1.6, 0.5, 0.0));

        var seen = new Dictionary<string, List<SpeakerSegmentUpdate>>();
        for (int offset = 0; offset < audio.Length; offset += 320)
        {
            var update = diarizer.Push(audio.AsSpan(offset, Math.Min(320, audio.Length - offset)));
            foreach (var segment in update.Segments)
            {
                if (!seen.TryGetValue(segment.Id, out var list)) seen[segment.Id] = list = [];
                list.Add(segment);
            }
        }
        foreach (var segment in diarizer.Flush().Segments)
        {
            if (!seen.TryGetValue(segment.Id, out var list)) seen[segment.Id] = list = [];
            list.Add(segment);
        }

        Assert.NotEmpty(seen);
        foreach (var (id, revisions) in seen)
        {
            Assert.Equal(Enumerable.Range(0, revisions.Count), revisions.Select(r => r.Revision));
            Assert.All(revisions.SkipLast(1), r => Assert.False(r.IsFinal, $"{id} was final before its last revision"));
            Assert.True(revisions[^1].IsFinal);
            Assert.All(revisions, r => Assert.Equal(revisions[0].Start, r.Start, 6));
        }
    }

    [Fact]
    public void Geometry_that_cannot_decide_its_own_region_is_rejected()
    {
        var segmentation = new ToneMaskSegmentation(WindowSamples);
        // Hop larger than a third of the window: the window cannot see the region it decides.
        var bad = Config() with { HopSamples = WindowSamples / 2 };
        Assert.Throws<ArgumentException>(() => new StreamingDiarizer(segmentation, new ToneEmbedding(), bad));

        // Hop that does not land the owned region on the frame grid.
        var offGrid = Config() with { HopSamples = 480 };
        Assert.Throws<ArgumentException>(() => new StreamingDiarizer(segmentation, new ToneEmbedding(), offGrid));
    }

    [Fact]
    public void A_rate_mismatch_is_rejected_at_construction()
    {
        var bad = Config() with { SampleRate = 8_000 };
        Assert.Throws<ArgumentException>(() => new StreamingDiarizer(new ToneMaskSegmentation(WindowSamples), new ToneEmbedding(), bad));
    }

    private static string[] Describe(IReadOnlyList<SpeakerSegmentUpdate> segments) =>
        segments.Select(s => $"{s.Id}|{s.Speaker}|{s.Start:F3}|{s.End:F3}|{s.IsFinal}").ToArray();
}
