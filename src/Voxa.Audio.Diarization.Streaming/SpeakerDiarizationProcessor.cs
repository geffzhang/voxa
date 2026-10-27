using System.Buffers.Binary;
using System.Threading.Channels;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Voxa.Audio.Diarization;
using Voxa.Frames;
using Voxa.Processors;

namespace Voxa.Audio.Diarization.Streaming;

/// <summary>
/// The streaming-diarization stage (VLS-011) — placed on the raw input, <b>before</b> the VAD, because diarization
/// needs the continuous stream (silence included) and the ungated audio is the only place it exists.
///
/// <para>It does two jobs:</para>
/// <list type="number">
/// <item><b>Owns the input audio clock.</b> Every <see cref="AudioRawFrame"/> it forwards is stamped with its absolute
/// start sample, so downstream stages (STT → transcript span, this stage's segments) share one timeline instead of two
/// clocks that have to be reconciled. Chunks replayed by the VAD's pre-roll keep their stamps, so a replayed chunk
/// still points at where it came from.</item>
/// <item><b>Feeds the incremental diarizer off the data path.</b> Audio is converted to float, copied into a bounded
/// queue, and consumed by a single worker task that calls <see cref="StreamingDiarizer.Push"/>. Model inference never
/// runs on the data loop: a slow window delays diarization (and is reported as a timeline gap once the queue
/// overflows), never the conversation.</item>
/// </list>
///
/// <para>Output is a stream of <see cref="SpeakerSegmentFrame"/> upserts plus <see cref="SpeakerMergedFrame"/> notices.
/// They ride the normal data channel (ordered control frames), so they cannot overtake the audio they describe.</para>
/// </summary>
public sealed class SpeakerDiarizationProcessor : FrameProcessor
{
    private readonly StreamingDiarizer _diarizer;
    private readonly SpeakerDiarizationOptions _options;
    private readonly ILogger _logger;

    private readonly Channel<float[]> _queue;
    private readonly CancellationTokenSource _workerCts = new();
    private Task? _worker;

    private long _nextSample;          // absolute start sample of the next audio chunk
    private long _droppedSamples;      // samples the queue refused; reported to the diarizer as a gap
    private bool _rateMismatchLogged;
    private bool _channelMismatchLogged;

    /// <summary>
    /// Build the stage from its model seams — the host registers <see cref="ISpeakerMaskSegmentation"/> and
    /// <see cref="ISpeakerEmbedding"/>, which is the same pair the batch pipeline composes.
    /// </summary>
    public SpeakerDiarizationProcessor(
        ISpeakerMaskSegmentation segmentation,
        ISpeakerEmbedding embedding,
        SpeakerDiarizationOptions? options = null,
        ILogger? logger = null)
        : this(new StreamingDiarizer(segmentation, embedding, (options ?? new SpeakerDiarizationOptions()).Diarizer),
               options, logger)
    {
    }

    /// <summary>Build the stage around an already-configured diarizer (a host may share its speaker state).</summary>
    public SpeakerDiarizationProcessor(
        StreamingDiarizer diarizer,
        SpeakerDiarizationOptions? options = null,
        ILogger? logger = null)
        : base("SpeakerDiarization")
    {
        _diarizer = diarizer ?? throw new ArgumentNullException(nameof(diarizer));
        _options = options ?? new SpeakerDiarizationOptions();
        _logger = logger ?? NullLogger.Instance;

        if (_options.AudioQueueCapacity <= 0)
            throw new ArgumentOutOfRangeException(nameof(options), "AudioQueueCapacity must be positive.");

        _queue = Channel.CreateBounded<float[]>(new BoundedChannelOptions(_options.AudioQueueCapacity)
        {
            FullMode = BoundedChannelFullMode.Wait, // never used: writes are TryWrite-only (see ProcessFrameAsync)
            SingleReader = true,
            SingleWriter = true,
        });
    }

    /// <summary>The diarizer this stage drives (exposed for hosts that query speaker state).</summary>
    public StreamingDiarizer Diarizer => _diarizer;

    /// <summary>Absolute sample position of the next chunk on the session's input timeline.</summary>
    public long SamplesSeen => Interlocked.Read(ref _nextSample);

    /// <summary>Samples the queue refused so far (each becomes a reported gap).</summary>
    public long DroppedSamples => Interlocked.Read(ref _droppedSamples);

    protected override ValueTask OnStartAsync(StartFrame frame, CancellationToken ct)
    {
        _worker = Task.Run(() => WorkerAsync(_workerCts.Token), CancellationToken.None);
        return ValueTask.CompletedTask;
    }

    protected override async ValueTask ProcessFrameAsync(Frame frame, CancellationToken ct)
    {
        if (frame is not AudioRawFrame audio)
        {
            await PushFrameAsync(frame, ct).ConfigureAwait(false);
            return;
        }

        long start = Interlocked.Read(ref _nextSample);
        int samples = audio.Pcm.Length / 2 / Math.Max(1, audio.Channels);
        long next = start + samples;
        Interlocked.Exchange(ref _nextSample, next);

        // Stamp before forwarding: with the clock on the frame, every downstream stage (and the transcript span) reads
        // the same timeline this stage feeds the diarizer.
        var stamped = audio with { StartSample = start };

        if (!Acceptable(audio)) { await PushFrameAsync(stamped, ct).ConfigureAwait(false); return; }

        var copy = ToFloat(audio.Pcm.Span, samples);
        if (!_queue.Writer.TryWrite(copy))
        {
            // Full queue: the diarizer is behind. Count the audio as a gap so its timeline stays true, and let the
            // conversation continue — blocking here would trade a diarization label for the user's answer.
            Interlocked.Add(ref _droppedSamples, samples);
        }

        await PushFrameAsync(stamped, ct).ConfigureAwait(false);
    }

    protected override async ValueTask OnEndAsync(EndFrame frame, CancellationToken ct)
    {
        await DrainAndFlushAsync(ct).ConfigureAwait(false);
    }

    protected override async ValueTask DisposeAsyncCore()
    {
        _queue.Writer.TryComplete();
        try { _workerCts.Cancel(); } catch (ObjectDisposedException) { }
        await WaitWorkerAsync(TimeSpan.FromSeconds(1)).ConfigureAwait(false);
        _workerCts.Dispose();
    }

    // ---------------------------------------------------------------- worker

    private async Task WorkerAsync(CancellationToken ct)
    {
        try
        {
            await foreach (var chunk in _queue.Reader.ReadAllAsync(ct).ConfigureAwait(false))
                await ProcessChunkAsync(chunk, ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException) { /* shutdown */ }
        catch (ChannelClosedException) { /* shutdown */ }
        catch (Exception ex)
        {
            _logger.LogError(ex, "SpeakerDiarization: incremental diarizer failed; further audio is not diarized");
        }
    }

    private async ValueTask ProcessChunkAsync(float[] chunk, CancellationToken ct)
    {
        long dropped = Interlocked.Exchange(ref _droppedSamples, 0);
        if (dropped > 0)
        {
            _logger.LogDebug("SpeakerDiarization: {Samples} samples were dropped by the queue; timeline gap reported", dropped);
            await PublishAsync(_diarizer.NotifyGap((int)Math.Min(dropped, int.MaxValue)), ct).ConfigureAwait(false);
        }
        await PublishAsync(_diarizer.Push(chunk), ct).ConfigureAwait(false);
    }

    private async ValueTask DrainAndFlushAsync(CancellationToken ct)
    {
        _queue.Writer.TryComplete();
        await WaitWorkerAsync(_options.FlushTimeout).ConfigureAwait(false);

        long dropped = Interlocked.Exchange(ref _droppedSamples, 0);
        if (dropped > 0) await PublishAsync(_diarizer.NotifyGap((int)Math.Min(dropped, int.MaxValue)), ct).ConfigureAwait(false);

        // Flushing runs the tail windows and commits every open segment; it happens once, at teardown, so the cost is
        // a bounded burst rather than a latency tax on the conversation.
        await PublishAsync(_diarizer.Flush(), ct).ConfigureAwait(false);
    }

    private async ValueTask WaitWorkerAsync(TimeSpan timeout)
    {
        if (_worker is null) return;
        try
        {
            await _worker.WaitAsync(timeout).ConfigureAwait(false);
        }
        catch (TimeoutException)
        {
            _logger.LogWarning(
                "SpeakerDiarization: worker did not drain within {Ms} ms; flushing what has been decided so far",
                (int)timeout.TotalMilliseconds);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "SpeakerDiarization: worker ended with an error during teardown");
        }
        _worker = null;
    }

    private async ValueTask PublishAsync(DiarizerUpdate update, CancellationToken ct)
    {
        foreach (var segment in update.Segments)
        {
            await PublishAsync(
                new SpeakerSegmentFrame(
                    segment.Id, segment.Speaker, segment.Start, segment.End, segment.IsFinal, segment.Revision, segment.Confidence),
                ct).ConfigureAwait(false);
        }

        foreach (var merge in update.Merges)
        {
            await PublishAsync(
                new SpeakerMergedFrame(merge.RetiredSpeaker, merge.SurvivingSpeaker, merge.AtSeconds), ct).ConfigureAwait(false);
        }
    }

    // The worker outlives individual data frames, so a closed downstream (teardown racing the tail) is normal, not an
    // error: swallow it rather than faulting a background task nobody awaits.
    private async ValueTask PublishAsync(Frame frame, CancellationToken ct)
    {
        try
        {
            await PushFrameAsync(frame, ct).ConfigureAwait(false);
        }
        catch (ChannelClosedException) { }
        catch (ObjectDisposedException) { }
        catch (OperationCanceledException) { }
    }

    // ---------------------------------------------------------------- audio helpers

    private bool Acceptable(AudioRawFrame audio)
    {
        if (audio.SampleRate != _diarizer.SampleRate)
        {
            if (!_rateMismatchLogged)
            {
                _rateMismatchLogged = true;
                _logger.LogWarning(
                    "SpeakerDiarization received {ActualRate} Hz audio but diarizes at {ExpectedRate} Hz; forwarding it " +
                    "on the input clock without diarizing for the rest of the session. Resample upstream.",
                    audio.SampleRate, _diarizer.SampleRate);
            }
            return false;
        }

        if (audio.Channels != 1)
        {
            if (!_channelMismatchLogged)
            {
                _channelMismatchLogged = true;
                _logger.LogWarning(
                    "SpeakerDiarization received {Channels}-channel audio; diarization needs mono. Forwarding it on the " +
                    "input clock without diarizing for the rest of the session.", audio.Channels);
            }
            return false;
        }

        return true;
    }

    // PCM16 LE → normalised float. One allocation per chunk (the queue owns it), which is the price of getting the
    // conversion off the worker: the data loop must not hold a reference to a frame's memory across frames.
    private static float[] ToFloat(ReadOnlySpan<byte> pcm, int samples)
    {
        var buffer = new float[samples];
        for (int i = 0; i < samples; i++)
            buffer[i] = BinaryPrimitives.ReadInt16LittleEndian(pcm.Slice(i * 2, 2)) / 32768f;
        return buffer;
    }
}
