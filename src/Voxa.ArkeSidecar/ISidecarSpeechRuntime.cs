namespace Voxa.ArkeSidecar;

internal interface ISidecarSpeechRuntime
{
    HealthResponse Health { get; }
    Task<byte[]> SynthesizeAsync(TtsRequest request, CancellationToken ct);
    Task<string> TranscribeAsync(ReadOnlyMemory<byte> audio, string? contentType, CancellationToken ct);
}

internal sealed class SidecarUnavailableException(string message) : Exception(message);
