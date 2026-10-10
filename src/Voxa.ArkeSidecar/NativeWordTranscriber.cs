using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using Whisper.net;
using Whisper.net.LibraryLoader;
using Voxa.Speech.WhisperCpp;

namespace Voxa.ArkeSidecar;

/// <summary>Uses acoustic DTW boundaries, never Whisper's heuristic token-duration interpolation.</summary>
internal sealed class NativeWordTranscriber : IAsyncDisposable
{
    private WhisperFactory? _factory;
    private WordTimingEngine? _identity;

    // Calls are serialized by SidecarSpeechRuntime's shared STT gate, including disposal.
    public async Task<WordTimingResponse> TranscribeAsync(string modelPath, byte[] pcm, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        if (pcm.Length == 0 || (pcm.Length & 1) != 0)
            throw new InvalidDataException("Word timing requires non-empty PCM16 audio.");
        if (pcm.Length < 6400)
            throw new WordTimingRejectedException("Word timing requires at least 0.2 seconds of audio.");
        if (_factory is null)
        {
            // The sidecar is CPU-only. Never replace the process's loaded native runtime.
            RuntimeOptions.RuntimeLibraryOrder = [RuntimeLibrary.Cpu, RuntimeLibrary.CpuNoAvx];
            using var model = File.OpenRead(modelPath);
            var digest = Convert.ToHexString(await SHA256.HashDataAsync(model, ct).ConfigureAwait(false)).ToLowerInvariant();
            ct.ThrowIfCancellationRequested();
            var heads = AlignmentHeads(digest);
            _factory = WhisperFactory.FromPath(modelPath, new WhisperFactoryOptions
            {
                UseGpu = false, UseDtwTimeStamps = true, HeadsPreset = heads,
            });
            _identity = new WordTimingEngine("whisper.cpp/dtw-word-boundaries-v1",
                typeof(WhisperFactory).Assembly.GetName().Version?.ToString() ?? "unknown", "sha256:" + digest);
        }
        var samples = new float[pcm.Length / 2];
        for (var i = 0; i < samples.Length; i++)
            samples[i] = BinaryPrimitives.ReadInt16LittleEndian(pcm.AsSpan(i * 2, 2)) / 32768f;
        var seconds = samples.Length / 16000d;
        var text = new StringBuilder();
        var tokens = new List<MeasuredToken>();
        foreach (var window in AlignmentWindows(samples))
        {
            ct.ThrowIfCancellationRequested();
            await using var processor = _factory.CreateBuilder().WithLanguage("en")
                .WithThreads(Math.Min(4, Environment.ProcessorCount)).WithProbabilities().Build();
            var offset = window.Start / 16000d;
            var windowTokens = new List<(string Text, double Start, double Probability)>();
            await foreach (var segment in processor.ProcessAsync(samples.AsSpan(window.Start, window.Length).ToArray(), ct).ConfigureAwait(false))
            {
                text.Append(' ').Append(segment.Text.Trim());
                foreach (var token in segment.Tokens.Where(token => !IsControl(token.Text ?? string.Empty)))
                    windowTokens.Add((token.Text ?? string.Empty, offset + token.DtwTimestamp / 100d, token.Probability));
            }
            for (var i = 0; i < windowTokens.Count; i++)
            {
                // Segment timestamp tokens can precede the final lexical DTW boundary.
                // Use consecutive DTW starts across segments, then the measured PCM window end.
                // The acoustic silence pass below tightens trailing quiet inside that interval.
                var token = windowTokens[i];
                var end = i + 1 < windowTokens.Count ? windowTokens[i + 1].Start : (window.Start + window.Length) / 16000d;
                tokens.Add(new MeasuredToken(token.Text, token.Start, end, token.Probability));
            }
        }
        ct.ThrowIfCancellationRequested();
        var words = TrimAtAcousticPause(ComposeWords(tokens, seconds), samples);
        if (text.ToString().Trim().Length > 0 && words.Count == 0)
            throw new WordTimingRejectedException("The recognizer returned text without usable word timings.");
        return new WordTimingResponse(text.ToString().Trim(), seconds, _identity!, words);
    }

    internal sealed record MeasuredToken(string Text, double Start, double End, double Probability);

    internal static IReadOnlyList<(int Start, int Length)> AlignmentWindows(float[] samples)
    {
        // The pinned native DTW callback drops later >30-second windows. Independent short
        // windows avoid that bug; split only inside measured silence, never inside a word.
        const int rate = 16000, maximum = 28 * rate, quietRequired = 1920, frame = 160;
        var windows = new List<(int, int)>();
        var start = 0;
        while (samples.Length - start > maximum)
        {
            var quietStart = -1;
            var split = -1;
            for (var from = start + rate; from + frame <= start + maximum; from += frame)
            {
                double energy = 0;
                for (var i = from; i < from + frame; i++) energy += (double)samples[i] * samples[i];
                if (energy / frame <= 0.0000001)
                {
                    if (quietStart < 0) quietStart = from;
                    if (from + frame - quietStart >= quietRequired) split = quietStart + (from + frame - quietStart) / 2;
                }
                else quietStart = -1;
            }
            if (split < 0) throw new WordTimingRejectedException("Long audio requires a measurable silent pause within each 28-second alignment window.");
            windows.Add((start, split - start));
            start = split;
        }
        windows.Add((start, samples.Length - start));
        return windows;
    }

    private static bool IsControl(string text) => string.IsNullOrWhiteSpace(text)
        || text.StartsWith("[_", StringComparison.Ordinal) || text.StartsWith("[[_", StringComparison.Ordinal)
        || text.StartsWith("<|", StringComparison.Ordinal);

    internal static WhisperAlignmentHeadsPreset AlignmentHeads(string digest)
    {
        // A filename is not a model identity; a wrong preset can crash native inference.
        // Limit the capability to Arke's provisioned and validated English model.
        if (WhisperCppModelCatalog.TryGet("base.en", out var artifact)
            && string.Equals(digest, artifact.Sha256, StringComparison.OrdinalIgnoreCase)) return WhisperAlignmentHeadsPreset.BaseEn;
        throw new SidecarUnavailableException("Word timing requires the provisioned base.en model.");
    }

    internal static IReadOnlyList<TimedWord> ComposeWords(IEnumerable<MeasuredToken> tokens, double seconds)
    {
        if (!double.IsFinite(seconds) || seconds <= 0) throw new WordTimingRejectedException("Invalid audio duration.");
        var result = new List<TimedWord>();
        var text = new StringBuilder();
        double start = 0, end = 0, probability = 1;
        void Flush()
        {
            if (text.Length == 0) return;
            if (end <= start || (result.Count > 0 && start < result[^1].EndSec))
                throw new WordTimingRejectedException($"The recognizer returned ambiguous or overlapping word bounds at word {result.Count + 1} ({start:R}–{end:R} seconds).");
            result.Add(new TimedWord(text.ToString(), start, end, probability));
            text.Clear();
        }
        foreach (var token in tokens)
        {
            var word = token.Text?.Trim() ?? string.Empty;
            if (IsControl(word)) continue;
            if (!double.IsFinite(token.Start) || !double.IsFinite(token.End) || !double.IsFinite(token.Probability)
                || token.Start < 0 || token.End < token.Start || token.End > seconds
                || token.Probability < 0 || token.Probability > 1)
                throw new WordTimingRejectedException($"The recognizer returned invalid token bounds or confidence ({token.Start:R}–{token.End:R} seconds, probability {token.Probability:R}, duration {seconds:R}).");
            // A token spanning whitespace cannot be divided without inventing separate bounds.
            if (word.Any(char.IsWhiteSpace))
                throw new WordTimingRejectedException("The recognizer returned multiple words under one timestamp.");
            var lexical = word.Any(char.IsLetterOrDigit);
            if (char.IsWhiteSpace(token.Text![0]) && lexical) Flush();
            if (text.Length == 0) { start = token.Start; end = token.End; probability = token.Probability; }
            else
            {
                if (token.Start < start || token.End < end)
                    throw new WordTimingRejectedException("The recognizer returned out-of-order tokens.");
                end = token.End;
                probability = Math.Min(probability, token.Probability);
            }
            text.Append(word);
        }
        Flush();
        return result;
    }

    internal static IReadOnlyList<TimedWord> TrimAtAcousticPause(IReadOnlyList<TimedWord> words, float[] samples)
    {
        const int rate = 16000, frame = 160, minimumQuiet = 1920; // 10 ms frames, 120 ms sustained silence
        const double quietMeanSquare = 0.0000001; // near-silence, about -70 dBFS
        return words.Select(word =>
        {
            var first = Math.Max(0, (int)Math.Ceiling(word.StartSec * rate));
            var originalEnd = Math.Min(samples.Length, (int)Math.Floor(word.EndSec * rate));
            var end = originalEnd;
            var quietStart = -1;
            for (var from = first; from < originalEnd; from += frame)
            {
                var until = Math.Min(originalEnd, from + frame);
                double energy = 0;
                for (var i = from; i < until; i++) energy += (double)samples[i] * samples[i];
                if (energy / (until - from) <= quietMeanSquare)
                {
                    if (quietStart < 0) quietStart = from;
                    if (until - quietStart >= minimumQuiet) { end = quietStart; break; }
                }
                else quietStart = -1;
            }
            // DTW's next-word boundary may fall after speech has resumed. Clear at the first
            // sustained acoustic pause inside the interval, even if noise follows that pause.
            // A word with no audible energy or an unresolved very long gap is not highlight-ready.
            var measuredEnd = end < originalEnd ? end / (double)rate : word.EndSec;
            if (measuredEnd <= word.StartSec || measuredEnd - word.StartSec > 2.5)
                throw new WordTimingRejectedException("A word has no usable acoustic interval or an unresolved pause.");
            return word with { EndSec = measuredEnd };
        }).ToArray();
    }

    public ValueTask DisposeAsync()
    {
        _factory?.Dispose();
        _factory = null;
        return ValueTask.CompletedTask;
    }
}
