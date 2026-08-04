using System.Net;
using System.Net.Sockets;
using System.Net.Http.Json;
using System.Text.Json;

namespace Voxa.ArkeSidecar.Tests;

public class SidecarApplicationTests
{
    private sealed class FakeRuntime : ISidecarSpeechRuntime
    {
        public HealthResponse Health { get; set; } = new(
            true, "1.2.3", 1, "x64", ["kokoro", "whisper"],
            new Dictionary<string, EngineHealth> { ["kokoro"] = new(true, null), ["whisper"] = new(true, null) }, null);
        public TtsRequest? Tts { get; private set; }
        public byte[] SttAudio { get; private set; } = [];

        public Task<byte[]> SynthesizeAsync(TtsRequest request, CancellationToken ct)
        {
            Tts = request;
            return Task.FromResult("RIFFxxxxWAVE"u8.ToArray());
        }

        public Task<string> TranscribeAsync(ReadOnlyMemory<byte> audio, string? contentType, CancellationToken ct)
        {
            SttAudio = audio.ToArray();
            return Task.FromResult("hello harbour");
        }
    }

    [Fact]
    public async Task Exposes_health_and_exact_arke_voice_shape()
    {
        var (app, client, _) = await StartAsync();
        await using (app)
        {
            using var health = JsonDocument.Parse(await client.GetStringAsync("/health"));
            Assert.True(health.RootElement.GetProperty("ok").GetBoolean());
            Assert.Equal(1, health.RootElement.GetProperty("protocolVersion").GetInt32());
            Assert.Equal("x64", health.RootElement.GetProperty("architecture").GetString());
            Assert.True(health.RootElement.GetProperty("engineStatus").GetProperty("kokoro").GetProperty("ready").GetBoolean());

            using var voices = JsonDocument.Parse(await client.GetStringAsync("/voices"));
            Assert.Equal(6, voices.RootElement.GetArrayLength());
            Assert.Equal("af_bella", voices.RootElement[0].GetProperty("id").GetString());
            Assert.Equal("kokoro", voices.RootElement[0].GetProperty("engine").GetString());
        }
    }

    [Fact]
    public async Task Rejects_non_json_tts_requests_with_415()
    {
        var (app, client, _) = await StartAsync();
        await using (app)
        {
            using var content = new StringContent("hello");
            var response = await client.PostAsync("/tts", content);
            Assert.Equal(HttpStatusCode.UnsupportedMediaType, response.StatusCode);
        }
    }

    [Fact]
    public async Task Rejects_unbounded_tts_speed_cardinality()
    {
        var options = SidecarOptions.Parse(["--host", "127.0.0.1", "--port", "5901"]);
        var runtime = new SidecarSpeechRuntime(options);
        await Assert.ThrowsAsync<ArgumentException>(() =>
            runtime.SynthesizeAsync(new TtsRequest { Voice = "af_bella", Text = "hello", Speed = 1.01 }, default));
        await runtime.DisposeAsync();
    }

    [Fact]
    public async Task Adapts_tts_and_stt_requests_without_logging_content()
    {
        var (app, client, fake) = await StartAsync();
        await using (app)
        {
            var tts = await client.PostAsJsonAsync("/tts", new { voice = "af_bella", text = "secret speech", speed = 1.2 });
            Assert.Equal("audio/wav", tts.Content.Headers.ContentType?.MediaType);
            Assert.Equal("secret speech", fake.Tts?.Text);

            using var audio = new ByteArrayContent([1, 0, 2, 0]);
            audio.Headers.ContentType = new("audio/pcm");
            var stt = await client.PostAsync("/stt", audio);
            Assert.Equal("hello harbour", (await stt.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("text").GetString());
            Assert.Equal([1, 0, 2, 0], fake.SttAudio);
        }
    }

    [Fact]
    public void Rejects_non_loopback_clients()
    {
        Assert.False(SidecarApplication.IsLoopback(IPAddress.Parse("10.0.0.2")));
        Assert.True(SidecarApplication.IsLoopback(IPAddress.Loopback));
    }

    [Fact]
    public async Task Missing_assets_start_and_report_safe_unavailable_reason()
    {
        var runtime = new SidecarSpeechRuntime(SidecarOptions.Parse(["--host", "127.0.0.1", "--port", "5901"]));
        var health = runtime.Health;
        Assert.True(health.Ok);
        Assert.Equal("Kokoro is unavailable. Whisper is unavailable.", health.UnavailableReason);
        Assert.DoesNotContain(Path.GetTempPath(), health.UnavailableReason, StringComparison.OrdinalIgnoreCase);
        await runtime.DisposeAsync();
    }

    [Fact]
    public async Task Kokoro_and_whisper_availability_are_independent()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"voxa-arke-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        var whisper = Path.Combine(dir, "whisper.bin");
        await File.WriteAllBytesAsync(whisper, [1]);
        var runtime = new SidecarSpeechRuntime(SidecarOptions.Parse([
            "--host", "127.0.0.1", "--port", "5901", "--whisper-model", whisper]));
        try
        {
            Assert.True(runtime.Health.Ok);
            Assert.Equal("Kokoro is unavailable.", runtime.Health.UnavailableReason);
        }
        finally
        {
            await runtime.DisposeAsync();
            Directory.Delete(dir, true);
        }
    }

    private static async Task<(WebApplication App, HttpClient Client, FakeRuntime Runtime)> StartAsync()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        var fake = new FakeRuntime();
        var app = SidecarApplication.Build(["--host", "127.0.0.1", "--port", port.ToString()], fake);
        await app.StartAsync();
        return (app, new HttpClient { BaseAddress = new Uri($"http://127.0.0.1:{port}") }, fake);
    }
}
