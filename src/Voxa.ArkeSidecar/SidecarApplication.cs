using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Http.Json;

namespace Voxa.ArkeSidecar;

internal static class SidecarApplication
{
    private const long MaxAudioBytes = 64 * 1024 * 1024;

    public static WebApplication Build(string[] args, ISidecarSpeechRuntime? runtime = null)
    {
        var options = SidecarOptions.Parse(args);
        var builder = WebApplication.CreateSlimBuilder(new WebApplicationOptions { Args = [] });
        // The sidecar has one endpoint contract. Inherited ASPNETCORE_/Kestrel configuration
        // must not be able to add a public listener beside it.
        builder.Configuration.Sources.Clear();
        builder.WebHost.ConfigureKestrel(k =>
        {
            k.AddServerHeader = false;
            k.Limits.MaxRequestBodySize = MaxAudioBytes;
            k.Listen(System.Net.IPAddress.Parse(options.Host), options.Port);
        });
        builder.Logging.ClearProviders();
        builder.Services.Configure<JsonOptions>(json =>
        {
            json.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
            json.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
        });
        builder.Services.AddSingleton(options);
        if (runtime is null) builder.Services.AddSingleton<ISidecarSpeechRuntime, SidecarSpeechRuntime>();
        else builder.Services.AddSingleton(runtime);

        var app = builder.Build();
        app.Use(async (context, next) =>
        {
            if (context.Connection.RemoteIpAddress is not null
                && !IsLoopback(context.Connection.RemoteIpAddress))
            {
                context.Response.StatusCode = StatusCodes.Status403Forbidden;
                return;
            }
            await next(context).ConfigureAwait(false);
        });

        app.MapGet("/health", (ISidecarSpeechRuntime speech) => Results.Json(speech.Health));
        app.MapGet("/voices", () => Results.Json(VoiceCatalog.Voices));
        app.MapPost("/tts", HandleTtsAsync);
        app.MapPost("/stt", (HttpRequest request, ISidecarSpeechRuntime speech, CancellationToken ct) =>
            HandleSttAsync(request, speech, ct, words: false));
        app.MapPost("/stt/words", (HttpRequest request, ISidecarSpeechRuntime speech, CancellationToken ct) =>
            HandleSttAsync(request, speech, ct, words: true));
        return app;
    }

    private static async Task<IResult> HandleTtsAsync(HttpRequest request, ISidecarSpeechRuntime speech, CancellationToken ct)
    {
        try
        {
            var input = await request.ReadFromJsonAsync<TtsRequest>(ct).ConfigureAwait(false);
            if (input is null) return Problem(StatusCodes.Status400BadRequest, "Invalid JSON request.");
            return Results.File(await speech.SynthesizeAsync(input, ct).ConfigureAwait(false), "audio/wav");
        }
        catch (JsonException) { return Problem(StatusCodes.Status400BadRequest, "Invalid JSON request."); }
        catch (InvalidOperationException) { return Problem(StatusCodes.Status415UnsupportedMediaType, "Content-Type must be application/json."); }
        catch (ArgumentException ex) { return Problem(StatusCodes.Status400BadRequest, ex.Message); }
        catch (SidecarUnavailableException ex) { return Problem(StatusCodes.Status503ServiceUnavailable, ex.Message); }
    }

    private static async Task<IResult> HandleSttAsync(HttpRequest request, ISidecarSpeechRuntime speech, CancellationToken ct, bool words)
    {
        try
        {
            if (request.ContentLength is > MaxAudioBytes)
                return Problem(StatusCodes.Status413PayloadTooLarge, "Audio body exceeds 64 MiB.");
            using var body = new MemoryStream();
            await request.Body.CopyToAsync(body, ct).ConfigureAwait(false);
            if (body.Length > MaxAudioBytes) return Problem(StatusCodes.Status413PayloadTooLarge, "Audio body exceeds 64 MiB.");
            if (words) return Results.Json(await speech.TranscribeWordsAsync(
                body.GetBuffer().AsMemory(0, checked((int)body.Length)), request.ContentType, ct).ConfigureAwait(false));
            var text = await speech.TranscribeAsync(
                body.GetBuffer().AsMemory(0, checked((int)body.Length)), request.ContentType, ct).ConfigureAwait(false);
            return Results.Json(new { text });
        }
        catch (InvalidDataException ex) { return Problem(StatusCodes.Status415UnsupportedMediaType, ex.Message); }
        catch (WordTimingRejectedException ex) { return Problem(StatusCodes.Status422UnprocessableEntity, ex.Message); }
        catch (SidecarUnavailableException ex) { return Problem(StatusCodes.Status503ServiceUnavailable, ex.Message); }
    }

    private static IResult Problem(int status, string detail) => Results.Problem(statusCode: status, detail: detail);
    internal static bool IsLoopback(System.Net.IPAddress address) => System.Net.IPAddress.IsLoopback(address);
}
