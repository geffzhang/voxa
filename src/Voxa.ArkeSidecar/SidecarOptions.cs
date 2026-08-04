using System.Net;

namespace Voxa.ArkeSidecar;

internal sealed record SidecarOptions(
    string Host,
    int Port,
    string? KokoroModel,
    string? KokoroConfig,
    string? KokoroVoices,
    string? WhisperModel,
    string? Espeak,
    string? EspeakData)
{
    public static SidecarOptions Parse(string[] args)
    {
        var values = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        for (var i = 0; i < args.Length; i += 2)
        {
            if (i + 1 >= args.Length || !args[i].StartsWith("--", StringComparison.Ordinal))
                throw new ArgumentException("Arguments must be discrete --name value pairs.");
            if (!KnownArguments.Contains(args[i]))
                throw new ArgumentException($"Unknown argument '{args[i]}'.");
            values[args[i]] = args[i + 1];
        }

        var host = values.GetValueOrDefault("--host") ?? "127.0.0.1";
        if (!IPAddress.TryParse(host, out var address) || !IPAddress.IsLoopback(address))
            throw new ArgumentException("--host must be a loopback IP address (127.0.0.1 or ::1).");

        var portText = values.GetValueOrDefault("--port") ?? Environment.GetEnvironmentVariable("PORT") ?? "5901";
        if (!int.TryParse(portText, out var port) || port is < 1 or > 65535)
            throw new ArgumentException("--port must be an integer from 1 through 65535.");

        return new SidecarOptions(
            address.ToString(), port,
            values.GetValueOrDefault("--kokoro-model"),
            values.GetValueOrDefault("--kokoro-config"),
            values.GetValueOrDefault("--kokoro-voices"),
            values.GetValueOrDefault("--whisper-model"),
            values.GetValueOrDefault("--espeak"),
            values.GetValueOrDefault("--espeak-data"));
    }

    private static readonly HashSet<string> KnownArguments = new(StringComparer.OrdinalIgnoreCase)
    {
        "--host", "--port", "--kokoro-model", "--kokoro-config", "--kokoro-voices",
        "--whisper-model", "--espeak", "--espeak-data",
    };
}
