namespace Voxa.Audio.Diarization;

/// <summary>
/// Streaming (online) diarization tunables (VLS-011). Two of these decide the latency/accuracy trade the
/// capability makes, so they are named explicitly rather than left to a default:
/// <see cref="WindowSamples"/> / <see cref="HopSamples"/> set the model's lookahead, and
/// <see cref="ContinuityWindow"/> decides how long a short or low-confidence turn may borrow the previous
/// speaker instead of creating a new one.
/// </summary>
public sealed record StreamingDiarizerConfig
{
    /// <summary>Sample rate of the audio pushed into the diarizer (must match the segmentation engine).</summary>
    public int SampleRate { get; init; } = 16_000;

    /// <summary>
    /// Model window length in samples. <c>0</c> = use the engine's <see cref="ISpeakerMaskSegmentation.WindowSamples"/>.
    /// Larger windows buy accuracy and cost lookahead: a segment can only be committed once the window that owns
    /// its centre has run, which lags "now" by <c>(window − hop) / 2</c>.
    /// </summary>
    public int WindowSamples { get; init; }

    /// <summary>
    /// Advance between successive windows. <c>0</c> = <c>WindowSamples / 10</c> (the reference's 10% hop). Must be
    /// at most a third of the window, or a window cannot see the region it is supposed to decide.
    /// </summary>
    public int HopSamples { get; init; }

    /// <summary>Probability at which a local speaker's track opens.</summary>
    public double Onset { get; init; } = 0.5;

    /// <summary>Probability under which the track starts counting toward closing (hysteresis below <see cref="Onset"/>).</summary>
    public double Offset { get; init; } = 0.35;

    /// <summary>How long a track must stay under <see cref="Offset"/> before the turn closes.</summary>
    public double MinSilenceDuration { get; init; } = 0.2;

    /// <summary>Turns shorter than this are not diarized on their own — they fall to the continuity policy.</summary>
    public double MinSpeechDuration { get; init; } = 0.25;

    /// <summary>Shorter spans are never embedded (the embedding would describe more silence than speech).</summary>
    public double MinEmbeddingDuration { get; init; } = 0.4;

    /// <summary>Cosine-distance ceiling for "this turn is a known speaker".</summary>
    public double MatchThreshold { get; init; } = 0.45;

    /// <summary>Cosine distance above which a <em>well-evidenced</em> turn starts a new speaker. Between this and
    /// <see cref="MatchThreshold"/> is the ambiguity band: attach to the nearest speaker, flag the segment provisional.</summary>
    public double NewSpeakerThreshold { get; init; } = 0.62;

    /// <summary>Speech seconds a single turn must contribute before it may create a speaker.</summary>
    public double MinNewSpeakerEvidence { get; init; } = 1.0;

    /// <summary>Mean per-frame activity a turn must reach before it may create a speaker.</summary>
    public double MinTurnConfidence { get; init; } = 0.5;

    /// <summary>How long after the previous turn a short/low-confidence turn may adopt that speaker.</summary>
    public double ContinuityWindow { get; init; } = 1.5;

    /// <summary>Same-speaker turns separated by at most this gap are merged into one segment.</summary>
    public double MaxMergeGap { get; init; } = 0.6;

    /// <summary>Hard speaker cap. Reaching it never renumbers anyone — new turns attach to the nearest speaker.</summary>
    public int MaxSpeakers { get; init; } = 10;

    /// <summary>Speech seconds after which a prototype stops moving (bounds long-session drift).</summary>
    public double PrototypeWeightCap { get; init; } = 20.0;

    /// <summary>How far back a speaker merge rewrites already-emitted segments. Older consumers remap via the merge event.</summary>
    public double RevisionWindow { get; init; } = 30.0;

    /// <summary>Two prototypes closer than this are the same voice; the younger id retires into the older one.</summary>
    public double MergeThreshold { get; init; } = 0.3;

    /// <summary>A speaker is mergeable only while its total evidence is below this (a well-settled speaker never disappears).</summary>
    public double MergeEvidence { get; init; } = 3.0;

    /// <summary>Bound on retained audio, in seconds — the ring buffer's capacity (never grows with the session).</summary>
    public double MaxBufferedSeconds { get; init; } = 60;

    /// <summary>Resolve the effective window length against the engine's geometry.</summary>
    public int ResolveWindowSamples(ISpeakerMaskSegmentation segmentation)
        => WindowSamples > 0 ? WindowSamples : segmentation.WindowSamples;

    /// <summary>Resolve the effective hop against the window (default: 10%).</summary>
    public int ResolveHopSamples(int windowSamples)
        => HopSamples > 0 ? HopSamples : Math.Max(1, windowSamples / 10);
}
