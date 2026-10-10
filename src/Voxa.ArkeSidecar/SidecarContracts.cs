using System.Runtime.InteropServices;
using System.Text.Json.Serialization;

namespace Voxa.ArkeSidecar;

internal sealed record LocalVoice(
    string Id,
    string Label,
    string Engine,
    string Language,
    string[] Attributes);

internal sealed record HealthResponse(
    bool Ok,
    string Version,
    int ProtocolVersion,
    string Architecture,
    string[] Engines,
    IReadOnlyDictionary<string, EngineHealth> EngineStatus,
    [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? UnavailableReason,
    EngineHealth? WordTiming = null)
{
    public static string CurrentArchitecture => RuntimeInformation.ProcessArchitecture switch
    {
        System.Runtime.InteropServices.Architecture.X64 => "x64",
        System.Runtime.InteropServices.Architecture.Arm64 => "arm64",
        _ => "unsupported",
    };
}

internal sealed record EngineHealth(bool Ready, [property: JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? Reason);

internal sealed record WordTimingEngine(string Id, string Version, string Model);
internal sealed record TimedWord(string Text, double StartSec, double EndSec, double Probability);
internal sealed record WordTimingResponse(string Text, double Seconds, WordTimingEngine Engine, IReadOnlyList<TimedWord> Words);

internal sealed class TtsRequest
{
    public string Voice { get; init; } = string.Empty;
    public string Text { get; init; } = string.Empty;
    public double? Speed { get; init; }
}

internal static class VoiceCatalog
{
    public static readonly LocalVoice[] Voices =
    [
        new("af_bella", "Bella", "kokoro", "en-US", ["warm", "low", "even", "female"]),
        new("af_nicole", "Nicole", "kokoro", "en-US", ["bright", "quick", "female"]),
        new("am_michael", "Michael", "kokoro", "en-US", ["deep", "steady", "male"]),
        new("am_adam", "Adam", "kokoro", "en-US", ["dry", "measured", "male"]),
        new("bf_emma", "Emma", "kokoro", "en-GB", ["cool", "precise", "female", "coastal"]),
        new("bm_george", "George", "kokoro", "en-GB", ["low", "gravel", "male", "weathered"]),
    ];

    public static bool Contains(string id) => Voices.Any(v => string.Equals(v.Id, id, StringComparison.Ordinal));
}
