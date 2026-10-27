using Voxa.Audio.Diarization;

namespace Voxa.Audio.Diarization.Streaming.Tests;

/// <summary>
/// A one-voice tone stream and the two model seams that recognise it: no model, no network, deterministic. The
/// segmentation seam can be gated so a test can hold the diarizer's worker inside inference and observe what the
/// data loop does while it is slow.
/// </summary>
internal static class MonoTone
{
    public const int SampleRate = 16_000;
    public const int ChunkSamples = 320; // 20 ms — the frame size the processor sees
    public const int WindowSamples = 3_200;
    public const int FrameSamples = 160;

    /// <summary>One <paramref name="seconds"/>-long 200 Hz tone as PCM16 LE, mono.</summary>
    public static byte[] Pcm(double seconds, double amplitude = 0.5)
    {
        int samples = (int)Math.Round(seconds * SampleRate);
        var pcm = new byte[samples * 2];
        for (int i = 0; i < samples; i++)
        {
            double value = amplitude * Math.Sin(2 * Math.PI * 200 * i / SampleRate);
            short pcmValue = (short)Math.Round(value * short.MaxValue);
            pcm[i * 2] = (byte)(pcmValue & 0xFF);
            pcm[i * 2 + 1] = (byte)((pcmValue >> 8) & 0xFF);
        }
        return pcm;
    }

    /// <summary>Energy at 200 Hz for one 10 ms frame, normalised by the frame length (threshold-friendly).</summary>
    public static double ToneEnergy(ReadOnlySpan<float> frame)
    {
        double w = 2 * Math.PI * 2 / frame.Length, coeff = 2 * Math.Cos(w), s1 = 0, s2 = 0;
        foreach (float sample in frame)
        {
            double s0 = sample + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        return Math.Max(0, s1 * s1 + s2 * s2 - coeff * s1 * s2) / (frame.Length * frame.Length);
    }
}

/// <summary>Reports one local speaker whose activity is "a tone is present in this frame".</summary>
internal sealed class TonePresenceSegmentation : ISpeakerMaskSegmentation
{
    public int SampleRate => MonoTone.SampleRate;
    public int WindowSamples => MonoTone.WindowSamples;
    public int FrameShiftSamples => MonoTone.FrameSamples;
    public int FrameOffsetSamples => 0;
    public int MaxLocalSpeakers => 1;

    /// <summary>Set to hold the diarizer's worker inside inference (the "slow model" case).</summary>
    public TaskCompletionSource? Gate { get; set; }

    public int Analyses { get; private set; }

    public SpeakerMaskWindow Analyze(ReadOnlySpan<float> window)
    {
        Analyses++;
        Gate?.Task.GetAwaiter().GetResult();

        int frames = window.Length / MonoTone.FrameSamples;
        var rows = new float[frames][];
        for (int f = 0; f < frames; f++)
        {
            double energy = MonoTone.ToneEnergy(window.Slice(f * MonoTone.FrameSamples, MonoTone.FrameSamples));
            rows[f] = [energy > 1e-4 ? 1f : 0f];
        }
        return new SpeakerMaskWindow(rows);
    }
}

/// <summary>One constant voice: every span is the same speaker, which is all a single-speaker stream needs.</summary>
internal sealed class ConstantEmbedding : ISpeakerEmbedding
{
    public int EmbeddingDim => 2;

    public int Calls { get; private set; }

    public float[] Embed(ReadOnlySpan<float> audio, int sampleRate)
    {
        Calls++;
        return [1f, 0f];
    }
}

internal static class StreamingConfig
{
    public static StreamingDiarizerConfig Diarizer => new()
    {
        SampleRate = MonoTone.SampleRate,
        WindowSamples = MonoTone.WindowSamples,
        HopSamples = 640,
        Onset = 0.5,
        Offset = 0.35,
        MinSilenceDuration = 0.04,
        MinSpeechDuration = 0.06,
        MinEmbeddingDuration = 0.08,
        MinNewSpeakerEvidence = 0.1,
        MinTurnConfidence = 0.5,
        MatchThreshold = 0.45,
        NewSpeakerThreshold = 0.62,
        ContinuityWindow = 0.3,
        MaxSpeakers = 4,
        MaxBufferedSeconds = 8,
    };
}
