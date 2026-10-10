using static Voxa.ArkeSidecar.NativeWordTranscriber;

namespace Voxa.ArkeSidecar.Tests;

public class NativeWordTranscriberTests(Xunit.Abstractions.ITestOutputHelper output)
{
    [Fact]
    public void Alignment_profile_uses_model_bytes_not_its_filename()
    {
        Assert.Equal(Whisper.net.WhisperAlignmentHeadsPreset.BaseEn,
            AlignmentHeads("a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002"));
        Assert.Throws<SidecarUnavailableException>(() => AlignmentHeads("unrecognized-model"));
    }

    [Fact]
    public void Joins_subwords_and_punctuation_without_redistributing_native_time()
    {
        var words = ComposeWords([
            new("[_BEG_]", -1, -1, 0),
            new(" Harbour", 0.18, 0.62, 0.95), new("master", 0.62, 0.92, 0.87),
            new(",", 0.92, 0.92, 0.99), new(" wait", 1.8, 2.16, 0.91), new("!", 2.16, 2.16, 0.98),
        ], 3);
        Assert.Equal([new TimedWord("Harbourmaster,", 0.18, 0.92, 0.87), new("wait!", 1.8, 2.16, 0.91)], words);
    }

    [Theory]
    [InlineData(-0.1, 0.2, 0.9)]
    [InlineData(0.4, 0.3, 0.9)]
    [InlineData(0.1, 3.1, 0.9)]
    [InlineData(0.1, 0.3, -0.1)]
    [InlineData(0.1, 0.3, 1.1)]
    [InlineData(double.NaN, 0.3, 0.9)]
    [InlineData(0.1, double.PositiveInfinity, 0.9)]
    public void Rejects_invalid_native_data(double start, double end, double probability)
        => Assert.Throws<WordTimingRejectedException>(() => ComposeWords([new(" word", start, end, probability)], 3));

    [Fact]
    public void Refuses_missing_overlapping_out_of_order_and_multiple_word_bounds()
    {
        Assert.Throws<WordTimingRejectedException>(() => ComposeWords([new(" word", 0.2, 0.2, 0.9)], 3));
        Assert.Throws<WordTimingRejectedException>(() => ComposeWords([new(" one", 0.1, 0.6, 0.9), new(" two", 0.5, 0.9, 0.9)], 3));
        Assert.Throws<WordTimingRejectedException>(() => ComposeWords([new(" fore", 0.2, 0.5, 0.9), new("ground", 0.1, 0.6, 0.9)], 3));
        Assert.Throws<WordTimingRejectedException>(() => ComposeWords([new(" two words", 0.1, 0.6, 0.9)], 3));
    }

    [Fact]
    public void Clears_a_measured_pause_without_guessing_word_lengths()
    {
        var samples = new float[48000];
        Array.Fill(samples, 0.1f, 0, 12000);
        Array.Fill(samples, 0.1f, 32000, 9600);
        var words = TrimAtAcousticPause([new("Hello", 0.1, 2.1, 0.9), new("again", 2.1, 2.6, 0.9)], samples);
        Assert.Equal(0.75, words[0].EndSec, 2);
        Assert.Equal(2.1, words[1].StartSec);
        Assert.Throws<WordTimingRejectedException>(() => TrimAtAcousticPause([new("hallucination", 0, 1, 0.9)], new float[16000]));
    }

    [Fact]
    public void Long_alignment_windows_split_only_inside_measured_silence()
    {
        var samples = Enumerable.Repeat(0.1f, 40 * 16000).ToArray();
        Assert.Throws<WordTimingRejectedException>(() => AlignmentWindows(samples));
        Array.Clear(samples, 20 * 16000, 16000);
        var windows = AlignmentWindows(samples);
        Assert.Equal(2, windows.Count);
        Assert.InRange(windows[0].Length, 20 * 16000, 21 * 16000);
        Assert.Equal(samples.Length, windows.Sum(window => window.Length));
        Assert.Equal(windows[0].Length, windows[1].Start);
    }

    [Fact]
    [Trait("Category", "LocalModels")]
    public async Task Base_model_returns_acoustic_word_bounds_for_real_speech()
    {
        var modelPath = Environment.GetEnvironmentVariable("VOXA_WORD_TIMING_TEST_MODEL");
        if (modelPath is null)
        {
            Voxa.Speech.WhisperCpp.WhisperCppModelCatalog.TryGet("base.en", out var model);
            var cache = new Voxa.Speech.VoxaModelCache(new(Voxa.Speech.VoxaModelCacheOptions.ResolveCacheRoot(), Offline: false));
            modelPath = await cache.ResolveAsync(model);
        }
        var wav = await File.ReadAllBytesAsync(Path.Combine(AppContext.BaseDirectory, "fixtures", "jfk.wav"));
        var pcm = AudioInput.ToWhisperPcm(wav, "audio/wav");
        await using var transcriber = new NativeWordTranscriber();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(60));
        var result = await transcriber.TranscribeAsync(modelPath, pcm, timeout.Token);
        Assert.Contains("country", result.Text, StringComparison.OrdinalIgnoreCase);
        Assert.True(result.Words.Count > 15);
        Assert.StartsWith("sha256:", result.Engine.Model);
        Assert.Equal("whisper.cpp/dtw-word-boundaries-v1", result.Engine.Id);
        Assert.Equal(pcm.Length / 32000d, result.Seconds);
        Assert.All(result.Words, word =>
        {
            Assert.True(word.EndSec > word.StartSec);
            Assert.InRange(word.EndSec, 0, result.Seconds);
            Assert.InRange(word.Probability, 0, 1);
        });
        // This noisy historic recording includes low-confidence words. Preserve the evidence
        // so the host can refuse highlighting; an endpoint result is not an approval to use it.
        Assert.Contains(result.Words, word => word.Probability < 0.5);
        for (var i = 1; i < result.Words.Count; i++) Assert.True(result.Words[i].StartSec >= result.Words[i - 1].EndSec);
        Assert.True(result.Words.Select(word => Math.Round(word.EndSec - word.StartSec, 2)).Distinct().Count() > 4);
        output.WriteLine(System.Text.Json.JsonSerializer.Serialize(result));
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => transcriber.TranscribeAsync(modelPath, pcm, cancelled.Token));

        // Narration passages commonly exceed Whisper's 30-second acoustic window.
        var longPcm = new byte[(pcm.Length + 32000) * 3];
        for (var i = 0; i < 3; i++) pcm.CopyTo(longPcm, i * (pcm.Length + 32000));
        using var longTimeout = new CancellationTokenSource(TimeSpan.FromSeconds(120));
        var longer = await transcriber.TranscribeAsync(modelPath, longPcm, longTimeout.Token);
        Assert.True(longer.Words.Count >= result.Words.Count * 3 - 3);
        Assert.True(longer.Words[^1].EndSec > 30);
        var firstPause = pcm.Length / 32000d + 0.5;
        Assert.DoesNotContain(longer.Words, word => word.StartSec <= firstPause && word.EndSec > firstPause);
    }
}
