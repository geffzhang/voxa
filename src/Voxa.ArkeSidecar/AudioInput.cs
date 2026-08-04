using NAudio.Wave;
using Voxa.Speech;

namespace Voxa.ArkeSidecar;

internal static class AudioInput
{
    public static byte[] ToWhisperPcm(ReadOnlySpan<byte> audio, string? contentType)
    {
        if (audio.IsEmpty) throw new InvalidDataException("Audio body is empty.");

        if (IsWav(contentType, audio))
        {
            var format = Pcm16Wav.FindData(audio);
            if (format.Format != 1 || format.SampleRate != 16000 || format.Channels != 1 || format.Bits != 16)
                throw new InvalidDataException("WAV input must be 16 kHz mono PCM16.");
            return audio.Slice(format.DataOffset, format.DataLength).ToArray();
        }

        if (contentType?.StartsWith("audio/webm", StringComparison.OrdinalIgnoreCase) == true)
            return DecodeWebM(audio);

        if (string.IsNullOrWhiteSpace(contentType)
            || contentType.StartsWith("application/octet-stream", StringComparison.OrdinalIgnoreCase)
            || contentType.StartsWith("audio/pcm", StringComparison.OrdinalIgnoreCase))
        {
            if ((audio.Length & 1) != 0) throw new InvalidDataException("PCM16 audio has an incomplete sample.");
            return audio.ToArray();
        }

        if (contentType.StartsWith("audio/L16", StringComparison.OrdinalIgnoreCase))
        {
            if ((audio.Length & 1) != 0) throw new InvalidDataException("PCM16 audio has an incomplete sample.");
            var littleEndian = audio.ToArray();
            for (var i = 0; i < littleEndian.Length; i += 2)
                (littleEndian[i], littleEndian[i + 1]) = (littleEndian[i + 1], littleEndian[i]);
            return littleEndian;
        }

        throw new InvalidDataException("Unsupported audio type; send 16 kHz mono PCM16 WAV or raw PCM16.");
    }

    private static byte[] DecodeWebM(ReadOnlySpan<byte> audio)
    {
        if (!OperatingSystem.IsWindows()) throw new InvalidDataException("WebM audio decoding is available on Windows only.");

        var path = Path.Combine(Path.GetTempPath(), $"voxa-arke-{Guid.NewGuid():N}.webm");
        try
        {
            File.WriteAllBytes(path, audio.ToArray());
            using var source = new MediaFoundationReader(path);
            using var resampler = new MediaFoundationResampler(source, new WaveFormat(16000, 16, 1));
            using var pcm = new MemoryStream();
            var buffer = new byte[81920];
            int read;
            while ((read = resampler.Read(buffer, 0, buffer.Length)) > 0) pcm.Write(buffer, 0, read);
            return pcm.ToArray();
        }
        catch (Exception ex) when (ex is not InvalidDataException)
        {
            throw new InvalidDataException("WebM audio could not be decoded.", ex);
        }
        finally
        {
            try { File.Delete(path); } catch { }
        }
    }

    private static bool IsWav(string? contentType, ReadOnlySpan<byte> audio)
        => contentType?.StartsWith("audio/wav", StringComparison.OrdinalIgnoreCase) == true
           || contentType?.StartsWith("audio/x-wav", StringComparison.OrdinalIgnoreCase) == true
           || (audio.Length >= 12 && audio[..4].SequenceEqual("RIFF"u8) && audio.Slice(8, 4).SequenceEqual("WAVE"u8));
}
