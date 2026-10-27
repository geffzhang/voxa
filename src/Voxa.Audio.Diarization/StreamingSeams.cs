namespace Voxa.Audio.Diarization;

/// <summary>
/// A <b>frame-level</b> speaker segmentation seam (VLS-011): one fixed-length window in, per-frame
/// <b>per-speaker</b> activity out. Unlike <see cref="ISpeakerSegmentation"/> — which answers "speech or silence"
/// for a whole recording — this keeps the powerset mask intact: several speakers may be active on the same frame
/// (overlap), and a speaker hand-over is two tracks rather than one on/off edge. That is what streaming
/// diarization needs to (a) cut turns at speaker changes, (b) keep overlapping speech, and (c) pick the audio
/// span to embed.
///
/// <para>The engine is deliberately <b>stateless and single-window</b>: window scheduling, the sliding-window
/// overlap, the absolute timeline, and all session state live in <see cref="StreamingDiarizer"/>, so the model
/// backend stays a pure function that is trivial to fake in tests.</para>
///
/// <para>Speaker indices here are <b>local to the window</b> (the powerset head's channel order) and carry no
/// identity across windows — <see cref="StreamingDiarizer"/> is what makes them global, by embedding each local
/// turn and matching it against the session's prototypes.</para>
/// </summary>
public interface ISpeakerMaskSegmentation
{
    /// <summary>The sample rate the engine requires; windows at any other rate are a programming error.</summary>
    int SampleRate { get; }

    /// <summary>Exact number of samples <see cref="Analyze"/> expects (the model's window geometry).</summary>
    int WindowSamples { get; }

    /// <summary>Hop between output frames, in samples (pyannote-3.0: 10 ms).</summary>
    int FrameShiftSamples { get; }

    /// <summary>
    /// Offset, in samples, from a frame's grid position to the point the frame describes (half the receptive
    /// field — the model centres a frame on the audio it summarises). Reported segment boundaries use it.
    /// </summary>
    int FrameOffsetSamples { get; }

    /// <summary>Width of each per-frame probability vector (the model's local speaker channel count).</summary>
    int MaxLocalSpeakers { get; }

    /// <summary>
    /// Score one window. <paramref name="window"/> must be exactly <see cref="WindowSamples"/> long.
    /// Returned probabilities are in [0, 1] per frame per local speaker; silence is all-zero, overlap is two
    /// (or more) non-zero entries on the same frame.
    /// </summary>
    SpeakerMaskWindow Analyze(ReadOnlySpan<float> window);
}

/// <summary>
/// One window's speaker masks (VLS-011): <c>FrameProbabilities[f][s]</c> is the probability that local speaker
/// <c>s</c> is active on frame <c>f</c>. Rows are fixed-width (<see cref="ISpeakerMaskSegmentation.MaxLocalSpeakers"/>).
/// </summary>
public sealed record SpeakerMaskWindow(IReadOnlyList<float[]> FrameProbabilities)
{
    /// <summary>Number of frames in the window.</summary>
    public int FrameCount => FrameProbabilities.Count;

    /// <summary>An empty window (no frames) — what a zero-length input decodes to.</summary>
    public static SpeakerMaskWindow Empty { get; } = new([]);
}
