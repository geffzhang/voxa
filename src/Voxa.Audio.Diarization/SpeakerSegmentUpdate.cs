namespace Voxa.Audio.Diarization;

/// <summary>
/// One publishing step for a speaker segment (VLS-011). Streaming diarization is <b>revising</b>, not
/// append-only: the model sees a window at a time, so a segment's end (and sometimes its speaker) is only known
/// once a later window has covered it. Consumers therefore treat
/// <see cref="SpeakerSegmentUpdate"/> as a keyed upsert on <see cref="Id"/>:
///
/// <list type="bullet">
/// <item><see cref="Revision"/> = 0 is the first publication of a segment; &gt; 0 replaces the state previously
/// published under the same <see cref="Id"/> (extended end, changed speaker after a merge, or the final verdict).</item>
/// <item><see cref="IsFinal"/> = <c>true</c> means the segment's geometry is settled — its end will not move. It is
/// the <i>commit</i> signal: a provisional segment is one that still reaches the window's commit frontier and may be
/// extended. (A speaker merge can still revise an already-final segment; that is reported by
/// <see cref="SpeakerMergeInfo"/> and bounded by the revision window, because it moves labels, not boundaries.)</item>
/// </list>
///
/// <para><see cref="Speaker"/> is a session-stable 0-based id (creation order); <see cref="UnknownSpeaker"/> marks a
/// span that is speech but could not be attributed (too short, too quiet, or an exhausted speaker cap). Times are
/// absolute seconds on the session's audio timeline — the same clock <c>TranscriptionFrame.StartTime</c> uses.</para>
///
/// <para><see cref="Confidence"/> is the strength of the <i>label</i>, not of the speech detection: 1.0 for a newly
/// created speaker, 1 − cosine distance for a matched one, roughly half that for a borrowed label (a short turn that
/// adopted the previous speaker, or an ambiguous one attached to its nearest), and 0 for
/// <see cref="UnknownSpeaker"/>. It never affects <see cref="IsFinal"/>, so a consumer that wants stricter labels can
/// filter on it without fighting the revision protocol.</para>
/// </summary>
public sealed record SpeakerSegmentUpdate(
    string Id,
    int Speaker,
    double Start,
    double End,
    bool IsFinal,
    int Revision,
    double Confidence)
{
    /// <summary>Speaker id for a span that is speech but not attributable.</summary>
    public const int UnknownSpeaker = -1;
}

/// <summary>
/// A speaker-merge notice (VLS-011): two prototypes turned out to be the same voice, so <see cref="RetiredSpeaker"/>
/// is retired into <see cref="SurvivingSpeaker"/>. Segments inside the config's revision window are re-published with
/// the surviving id; anything older is not rewritten, so consumers holding long-lived history remap it with this
/// record. Ids are never reused — a retired id stays retired for the session.
/// </summary>
public sealed record SpeakerMergeInfo(int RetiredSpeaker, int SurvivingSpeaker, double AtSeconds);

/// <summary>
/// Everything one diarizer step produced (VLS-011): segment upserts in timeline order plus any merges. Empty for
/// the (common) steps that did not commit anything — the type is a value so a no-op step allocates nothing new
/// beyond the wrapper.
/// </summary>
public sealed record DiarizerUpdate(
    IReadOnlyList<SpeakerSegmentUpdate> Segments,
    IReadOnlyList<SpeakerMergeInfo> Merges)
{
    /// <summary>A step that produced nothing.</summary>
    public static DiarizerUpdate Empty { get; } = new([], []);
}
