using Voxa.Audio.Diarization;

namespace Voxa.Audio.Diarization.Streaming;

/// <summary>
/// Tuning for the two streaming-diarization stages (VLS-011). <see cref="Diarizer"/> carries the model/window and
/// speaker-identity knobs; the rest is pipeline plumbing — how much audio may queue before the diarizer counts a gap,
/// and how long teardown waits for the tail to flush.
/// </summary>
public sealed record SpeakerDiarizationOptions
{
    /// <summary>Incremental diarizer configuration (window, hop, thresholds, caps).</summary>
    public StreamingDiarizerConfig Diarizer { get; init; } = new();

    /// <summary>
    /// Bounded queue depth in audio chunks (default 256 ≈ 5 s at 20 ms frames). The data loop never blocks on
    /// inference: when the queue is full the incoming chunk is counted as a gap in the timeline instead. Raise it on
    /// slow hardware rather than shortening the window.
    /// </summary>
    public int AudioQueueCapacity { get; init; } = 256;

    /// <summary>How long <c>EndFrame</c> waits for the worker to drain before flushing what it has.</summary>
    public TimeSpan FlushTimeout { get; init; } = TimeSpan.FromSeconds(5);

    /// <summary>Insert the transcript-attribution stage after STT. Off ⇒ segments are published, transcripts are untouched.</summary>
    public bool AttributeTranscripts { get; init; } = true;

    /// <summary>Attribution policy (see <see cref="SpeakerAttributionOptions"/>).</summary>
    public SpeakerAttributionOptions Attribution { get; init; } = new();
}

/// <summary>
/// How a transcript is matched to a speaker (VLS-011). The default is "greatest overlap, then continuity": a
/// transcript that spans a speaker change is attributed to whoever covers most of it, and a transcript whose span the
/// diarizer has not committed yet (it lags by the model's lookahead) borrows the speaker who was talking immediately
/// before it.
/// </summary>
public sealed record SpeakerAttributionOptions
{
    /// <summary>
    /// How far before a transcript's start a settled segment may sit and still lend its speaker. Covers the
    /// diarizer's lookahead when the transcript arrives first — the label is then corrected by the segment stream.
    /// </summary>
    public double ContinuityWindow { get; init; } = 1.5;

    /// <summary>Upper bound on segments held for lookup (oldest are dropped; a long call cannot grow this).</summary>
    public int MaxTrackedSegments { get; init; } = 512;

    /// <summary>Prefix used to spell a speaker id on the frame, e.g. <c>speaker 1</c>. Matches the batch CLI's format.</summary>
    public string SpeakerIdPrefix { get; init; } = "speaker ";
}
