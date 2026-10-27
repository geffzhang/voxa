namespace Voxa.Audio.Diarization;

/// <summary>
/// Fixed-capacity circular audio buffer addressed by <b>absolute</b> sample index (VLS-011) — the streaming
/// diarizer's only audio state. Absolute addressing is what makes the timeline survive eviction: a segment, a
/// window and an embedding span are all expressed as sample indices from the session start, so dropping old audio
/// changes what can be re-read, never what a timestamp means. Oldest samples are overwritten once the capacity is
/// reached, so memory is bounded by <c>capacity</c> for a session of any length.
/// </summary>
internal sealed class AudioRingBuffer
{
    private readonly float[] _buffer;
    private long _oldest; // absolute index of the oldest retained sample
    private long _end;    // absolute index one past the newest sample

    public AudioRingBuffer(int capacitySamples)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacitySamples);
        _buffer = new float[capacitySamples];
    }

    /// <summary>Absolute index one past the newest sample written.</summary>
    public long End => _end;

    /// <summary>Absolute index of the oldest sample still readable.</summary>
    public long Oldest => _oldest;

    /// <summary>Samples currently retained (never exceeds the capacity).</summary>
    public long Count => _end - _oldest;

    /// <summary>Capacity in samples.</summary>
    public int Capacity => _buffer.Length;

    /// <summary>Append samples, evicting the oldest ones when the buffer is full.</summary>
    public void Write(ReadOnlySpan<float> samples)
    {
        if (samples.Length == 0) return;

        int cap = _buffer.Length;
        if (samples.Length >= cap)
        {
            // A single write larger than the buffer: keep only its tail, which is all that can be retained.
            samples[^cap..].CopyTo(_buffer);
            _end += samples.Length;
            _oldest = _end - cap;
            return;
        }

        int startIndex = (int)(_end % cap);
        int first = Math.Min(samples.Length, cap - startIndex);
        samples[..first].CopyTo(_buffer.AsSpan(startIndex, first));
        if (first < samples.Length) samples[first..].CopyTo(_buffer);

        _end += samples.Length;
        if (_end - _oldest > cap) _oldest = _end - cap;
    }

    /// <summary>
    /// Copy the absolute range starting at <paramref name="absoluteStart"/> into <paramref name="destination"/>.
    /// A range that reaches past the newest sample, or starts before the oldest retained one, is zero-filled there
    /// (the caller is asking for a padded window); the return value is how many real samples were written.
    /// </summary>
    public int CopyTo(long absoluteStart, Span<float> destination)
    {
        destination.Clear();
        long from = Math.Max(absoluteStart, _oldest);
        long to = Math.Min(absoluteStart + destination.Length, _end);
        if (to <= from) return 0;

        int count = (int)(to - from);
        int offset = (int)(from - absoluteStart);
        int startIndex = (int)(from % _buffer.Length);
        int first = Math.Min(count, _buffer.Length - startIndex);
        _buffer.AsSpan(startIndex, first).CopyTo(destination.Slice(offset, first));
        if (count > first) _buffer.AsSpan(0, count - first).CopyTo(destination.Slice(offset + first, count - first));
        return count;
    }

    /// <summary>Drop everything older than <paramref name="absoluteIndex"/> (clamped to what is retained).</summary>
    public void DiscardBefore(long absoluteIndex)
    {
        if (absoluteIndex > _oldest) _oldest = Math.Min(absoluteIndex, _end);
    }
}
