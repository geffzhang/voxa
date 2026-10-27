using Voxa.Audio.Diarization;
using Voxa.Frames;
using Voxa.Processors;

namespace Voxa.Audio.Diarization.Streaming;

/// <summary>
/// Fills <see cref="TranscriptionFrame.SpeakerId"/> from the diarizer's segment stream (VLS-011) — the stage that turns
/// "who spoke when" into "who said this".
///
/// <para>It sits downstream of STT and consumes two things on the same channel: <see cref="TranscriptionFrame"/>s
/// (which carry their utterance's span in absolute seconds, stamped by the STT processor from the audio clock) and
/// <see cref="SpeakerSegmentFrame"/>s (the diarizer's upsert stream). Attribution is an interval join on that one
/// timeline: greatest overlap wins, ties go to the earlier-starting segment, and a span the diarizer has not committed
/// yet borrows the speaker who was talking immediately before it — the honest answer given that the model's lookahead
/// puts the diarizer behind the transcript. The segment stream remains the authoritative record: a later revision
/// carries the corrected label and the same clock, so a consumer can re-join history at any time.</para>
///
/// <para>It never blocks and never waits for the diarizer: a transcript that arrives before its segment gets the best
/// available label now, and the pipeline keeps its latency budget. An engine that already labels speakers (a
/// diarizing STT) is left alone — an existing <see cref="TranscriptionFrame.SpeakerId"/> is never overwritten.</para>
/// </summary>
public sealed class SpeakerAttributionProcessor : FrameProcessor
{
    private readonly SpeakerTimeline _timeline;
    private readonly SpeakerAttributionOptions _options;

    public SpeakerAttributionProcessor(SpeakerAttributionOptions? options = null)
        : base("SpeakerAttribution")
    {
        _options = options ?? new SpeakerAttributionOptions();
        _timeline = new SpeakerTimeline(_options);
    }

    /// <summary>Segments currently tracked (test seam for the bound).</summary>
    internal int TrackedSegments => _timeline.Count;

    protected override ValueTask ProcessFrameAsync(Frame frame, CancellationToken ct)
    {
        switch (frame)
        {
            case SpeakerSegmentFrame segment:
                _timeline.Upsert(segment);
                break;

            case SpeakerMergedFrame merge:
                _timeline.Merge(merge.RetiredSpeaker, merge.SurvivingSpeaker);
                break;

            // Only a transcript that carries its span can be joined; only one without a speaker is ours to label.
            case TranscriptionFrame { StartTime: { } start, EndTime: { } end, SpeakerId: null } transcript:
                int speaker = _timeline.Attribute(start, end);
                if (speaker != SpeakerSegmentUpdate.UnknownSpeaker)
                    frame = transcript with { SpeakerId = _options.SpeakerIdPrefix + speaker };
                break;
        }

        return PushFrameAsync(frame, ct);
    }
}
