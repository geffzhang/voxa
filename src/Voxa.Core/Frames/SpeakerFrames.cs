namespace Voxa.Frames;

/// <summary>
/// One speaker segment as published by streaming diarization (VLS-011) — a span of the input audio attributed to one
/// speaker. Ordered with the data stream (<see cref="ControlFrame"/>) so it cannot overtake the transcript it
/// explains, but it is <b>not</b> a system frame: a segment is information, not a preemption.
///
/// <para>The frame is a keyed upsert on <see cref="SegmentId"/>. <see cref="Revision"/> 0 is the first publication;
/// a higher revision replaces what was published for the same id (an extended end, or a re-label after a speaker
/// merge). <see cref="IsFinal"/> means the segment's <i>geometry</i> is settled — its end will not move — so a
/// consumer that only wants stable captions can ignore provisional ones. <see cref="Confidence"/> scores the label
/// itself (see <c>SpeakerSegmentUpdate.Confidence</c>): <c>0</c> for a span that is speech but not attributable,
/// <c>Speaker</c> = <c>-1</c>.</para>
///
/// <para>Times are absolute session seconds — the same clock <see cref="TranscriptionFrame.StartTime"/> carries, so
/// a transcript and a segment can be joined by interval overlap with no per-stream offset bookkeeping.</para>
/// </summary>
public sealed record SpeakerSegmentFrame(
    string SegmentId,
    int Speaker,
    double Start,
    double End,
    bool IsFinal,
    int Revision,
    double Confidence) : ControlFrame;

/// <summary>
/// A speaker id was retired into another (VLS-011): two voice prototypes turned out to be one person, so
/// <see cref="RetiredSpeaker"/> now means <see cref="SurvivingSpeaker"/>. Segments inside the diarizer's revision
/// window were already re-published with the surviving id; this frame is for consumers holding older history, which
/// cannot be rewritten in place. Ids are never reused, so a stale id always maps through this notice.
/// </summary>
public sealed record SpeakerMergedFrame(
    int RetiredSpeaker,
    int SurvivingSpeaker,
    double AtSeconds) : ControlFrame;
