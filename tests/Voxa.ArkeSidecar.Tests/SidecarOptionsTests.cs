namespace Voxa.ArkeSidecar.Tests;

public class SidecarOptionsTests
{
    [Theory]
    [InlineData("0.0.0.0")]
    [InlineData("192.168.1.10")]
    [InlineData("localhost")]
    public void Rejects_non_loopback_or_hostname(string host)
        => Assert.Throws<ArgumentException>(() => SidecarOptions.Parse(["--host", host]));

    [Fact]
    public void Preserves_discrete_paths_with_spaces()
    {
        var options = SidecarOptions.Parse([
            "--host", "127.0.0.1", "--port", "5901",
            "--kokoro-model", @"C:\Models With Spaces\model.onnx",
            "--kokoro-config", @"C:\Models With Spaces\config.json",
            "--kokoro-voices", @"C:\Models With Spaces\voices",
            "--whisper-model", @"C:\Models With Spaces\whisper.bin",
            "--espeak", @"C:\Program Files\espeak-ng.exe",
            "--espeak-data", @"C:\Program Files\espeak-ng-data"]);

        Assert.Equal(@"C:\Models With Spaces\model.onnx", options.KokoroModel);
        Assert.Equal(5901, options.Port);
    }
}
