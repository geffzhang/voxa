using Voxa.Speech;

namespace Voxa.ArkeSidecar.Tests;

public class AudioInputTests
{
    [Fact]
    public void Extracts_canonical_whisper_wav()
    {
        byte[] pcm = [1, 0, 2, 0];
        Assert.Equal(pcm, AudioInput.ToWhisperPcm(Pcm16Wav.Wrap(pcm, 16000), "audio/wav"));
    }

    [Fact]
    public void Rejects_wrong_wav_format()
    {
        var wav = Pcm16Wav.Wrap([1, 0], 24000);
        Assert.Throws<InvalidDataException>(() => AudioInput.ToWhisperPcm(wav, "audio/wav"));
    }

    [Fact]
    public void Converts_network_order_audio_l16_to_little_endian_pcm()
        => Assert.Equal([2, 1, 4, 3], AudioInput.ToWhisperPcm([1, 2, 3, 4], "audio/L16"));
}
