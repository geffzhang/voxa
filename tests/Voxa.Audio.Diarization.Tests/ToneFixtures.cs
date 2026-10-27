using Voxa.Audio.Diarization;

namespace Voxa.Audio.Diarization.Tests;

/// <summary>
/// Synthetic two-voice audio for streaming-diarization tests: "speaker A" is a 200 Hz tone, "speaker B" a 400 Hz one.
/// Both the fake segmentation engine and the fake embedder read those tones back out, so the test pair behaves like a
/// real model pair (one audio, one truth) without a model, a download, or a network.
/// </summary>
internal static class ToneSignal
{
    public const int SampleRate = 16_000;
    public const int FrameSamples = 160; // 10 ms — the models' frame grid
    public const int ToneABin = 2;       // 200 Hz at a 160-sample transform
    public const int ToneBBin = 4;       // 400 Hz

    /// <summary>Build <paramref name="seconds"/> of audio: each part adds its two tone amplitudes over its span.</summary>
    public static float[] Build(double seconds, params (double Start, double End, double A, double B)[] parts)
    {
        var audio = new float[(int)Math.Round(seconds * SampleRate)];
        foreach (var (start, end, ampA, ampB) in parts)
        {
            int from = Math.Clamp((int)Math.Round(start * SampleRate), 0, audio.Length);
            int to = Math.Clamp((int)Math.Round(end * SampleRate), from, audio.Length);
            for (int i = from; i < to; i++)
            {
                double t = i / (double)SampleRate;
                audio[i] += (float)(ampA * Math.Sin(2 * Math.PI * 200 * t) + ampB * Math.Sin(2 * Math.PI * 400 * t));
            }
        }
        return audio;
    }

    /// <summary>Goertzel power at <paramref name="bin"/> for one frame (integer bins ⇒ the two tones are orthogonal).</summary>
    public static double Power(ReadOnlySpan<float> frame, int bin)
    {
        double w = 2 * Math.PI * bin / frame.Length;
        double coeff = 2 * Math.Cos(w);
        double s1 = 0, s2 = 0;
        foreach (float sample in frame)
        {
            double s0 = sample + coeff * s1 - s2;
            s2 = s1;
            s1 = s0;
        }
        return Math.Max(0, s1 * s1 + s2 * s2 - coeff * s1 * s2);
    }

    /// <summary>Per-frame normalised tone energies, or all-zero for a silent frame.</summary>
    public static void FrameProbabilities(ReadOnlySpan<float> frame, Span<float> destination)
    {
        double a = Power(frame, ToneABin);
        double b = Power(frame, ToneBBin);
        double total = a + b;
        if (total <= 1e-6) { destination.Clear(); return; }
        destination[0] = (float)(a / total);
        destination[1] = (float)(b / total);
    }
}

/// <summary>
/// A stand-in for a segmentation model: reports, per frame, how much of the 200 Hz voice and the 400 Hz voice it hears.
/// Two non-zero channels on a frame is overlapping speech.
/// </summary>
internal sealed class ToneMaskSegmentation : ISpeakerMaskSegmentation
{
    public ToneMaskSegmentation(int windowSamples) => WindowSamples = windowSamples;

    public int SampleRate => ToneSignal.SampleRate;
    public int WindowSamples { get; }
    public int FrameShiftSamples => ToneSignal.FrameSamples;
    public int FrameOffsetSamples => 0;
    public int MaxLocalSpeakers => 2;

    public SpeakerMaskWindow Analyze(ReadOnlySpan<float> window)
    {
        if (window.Length != WindowSamples)
            throw new ArgumentException($"Expected {WindowSamples} samples; got {window.Length}.");

        int frames = window.Length / FrameShiftSamples;
        var rows = new float[frames][];
        for (int f = 0; f < frames; f++)
        {
            var row = new float[MaxLocalSpeakers];
            ToneSignal.FrameProbabilities(window.Slice(f * FrameShiftSamples, FrameShiftSamples), row);
            rows[f] = row;
        }
        return new SpeakerMaskWindow(rows);
    }
}

/// <summary>
/// A stand-in for a speaker-embedding model: the mean tone profile of the span, averaged over the frames where one
/// voice dominates (an embedding of a mixed overlap would describe neither speaker, which is not what a real model
/// does).
/// </summary>
internal sealed class ToneEmbedding : ISpeakerEmbedding
{
    public int EmbeddingDim => 2;

    /// <summary>How many spans were embedded — lets tests assert embedding work stays proportional to turns.</summary>
    public int Calls { get; private set; }

    public float[] Embed(ReadOnlySpan<float> audio, int sampleRate)
    {
        Calls++;
        var dominant = new double[2];
        var all = new double[2];
        var row = new float[2];
        int dominantFrames = 0;

        for (int offset = 0; offset + ToneSignal.FrameSamples <= audio.Length; offset += ToneSignal.FrameSamples)
        {
            ToneSignal.FrameProbabilities(audio.Slice(offset, ToneSignal.FrameSamples), row);
            all[0] += row[0];
            all[1] += row[1];
            if (Math.Max(row[0], row[1]) >= 0.7)
            {
                dominant[0] += row[0];
                dominant[1] += row[1];
                dominantFrames++;
            }
        }

        var vector = dominantFrames > 0 ? dominant : all;
        double norm = Math.Sqrt(vector[0] * vector[0] + vector[1] * vector[1]);
        return norm <= 0 ? [0f, 0f] : [(float)(vector[0] / norm), (float)(vector[1] / norm)];
    }
}

/// <summary>Applies a diarizer's upsert stream into the current state, the way a consumer would.</summary>
internal sealed class SegmentTable
{
    private readonly Dictionary<string, SpeakerSegmentUpdate> _byId = [];
    private readonly List<SpeakerMergeInfo> _merges = [];

    public void Apply(DiarizerUpdate update)
    {
        foreach (var segment in update.Segments) _byId[segment.Id] = segment;
        _merges.AddRange(update.Merges);
    }

    public IReadOnlyList<SpeakerSegmentUpdate> All =>
        _byId.Values.OrderBy(s => s.Start).ThenBy(s => s.Id, StringComparer.Ordinal).ToList();

    public IReadOnlyList<SpeakerSegmentUpdate> Final => All.Where(s => s.IsFinal).ToList();

    public IReadOnlyList<SpeakerMergeInfo> Merges => _merges;

    public SpeakerSegmentUpdate? Covering(double seconds) =>
        All.FirstOrDefault(s => seconds >= s.Start && seconds < s.End);
}
