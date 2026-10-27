using Voxa.Audio.Diarization;
using Voxa.Frames;
using Voxa.Processors;

namespace Voxa.Audio.Diarization.Streaming.Tests;

/// <summary>
/// The streaming-diarization stage on a real frame pipeline (VLS-011): the input clock is stamped on every forwarded
/// chunk, model inference stays off the data loop, a stalled queue becomes a timeline gap instead of backpressure,
/// and teardown commits what was still provisional.
/// </summary>
public class SpeakerDiarizationProcessorTests
{
    /// <summary>Sink that records what the stage published.</summary>
    private sealed class Collector : FrameProcessor
    {
        public Collector() : base("collector") { }

        public List<Frame> Frames { get; } = [];

        protected override ValueTask ProcessFrameAsync(Frame frame, CancellationToken ct)
        {
            lock (Frames) Frames.Add(frame);
            return ValueTask.CompletedTask;
        }

        public Frame[] Snapshot()
        {
            lock (Frames) return [.. Frames];
        }
    }

    private static SpeakerDiarizationOptions Options(
        int queueCapacity = 256,
        double maxBufferedSeconds = 8,
        TimeSpan? flushTimeout = null)
        => new()
        {
            Diarizer = StreamingConfig.Diarizer with { MaxBufferedSeconds = maxBufferedSeconds },
            AudioQueueCapacity = queueCapacity,
            FlushTimeout = flushTimeout ?? TimeSpan.FromSeconds(5),
        };

    private static AudioRawFrame Audio(byte[] pcm, int rate = MonoTone.SampleRate, int channels = 1)
        => new(pcm, rate, channels);

    private static async Task WaitForAsync(Collector sink, Func<Frame[], bool> condition, string because)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(5);
        while (DateTime.UtcNow < deadline)
        {
            if (condition(sink.Snapshot())) return;
            await Task.Delay(5);
        }
        throw new TimeoutException($"condition not met: {because}");
    }

    private static async Task WaitUntilAsync(Func<bool> condition, string because)
    {
        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(5);
        while (DateTime.UtcNow < deadline)
        {
            if (condition()) return;
            await Task.Delay(5);
        }
        throw new TimeoutException($"condition not met: {because}");
    }

    private static async Task PushAsync(FrameProcessor processor, byte[] pcm, int chunkSamples = MonoTone.ChunkSamples)
    {
        for (int offset = 0; offset < pcm.Length; offset += chunkSamples * 2)
        {
            int length = Math.Min(chunkSamples * 2, pcm.Length - offset);
            await processor.QueueFrameAsync(Audio(pcm.AsSpan(offset, length).ToArray()));
        }
    }

    // A session opens with StartFrame; the worker that drains the audio queue is started there, exactly like the
    // other stages that own a background pump.
    private static ValueTask BeginAsync(FrameProcessor processor) => processor.QueueFrameAsync(new StartFrame());

    [Fact]
    public async Task Every_forwarded_chunk_carries_its_absolute_start_sample()
    {
        await using var processor = new SpeakerDiarizationProcessor(
            new TonePresenceSegmentation(), new ConstantEmbedding(), Options());
        await using var sink = new Collector();
        processor.Link(sink);
        processor.Start();
        sink.Start();

        await BeginAsync(processor);
        await PushAsync(processor, MonoTone.Pcm(0.2)); // 10 chunks
        await WaitForAsync(sink, f => f.OfType<AudioRawFrame>().Count() == 10, "all chunks forwarded");

        var starts = sink.Snapshot().OfType<AudioRawFrame>().Select(a => a.StartSample).ToArray();
        Assert.Equal(Enumerable.Range(0, 10).Select(i => (long?)(i * MonoTone.ChunkSamples)), starts);
    }

    [Fact]
    public async Task Segments_are_published_and_committed_at_end()
    {
        await using var processor = new SpeakerDiarizationProcessor(
            new TonePresenceSegmentation(), new ConstantEmbedding(), Options());
        await using var sink = new Collector();
        processor.Link(sink);
        processor.Start();
        sink.Start();

        await BeginAsync(processor);
        await PushAsync(processor, MonoTone.Pcm(1.5));
        await processor.QueueFrameAsync(new EndFrame());

        await WaitForAsync(
            sink, f => f.OfType<SpeakerSegmentFrame>().Any(s => s.IsFinal), "a committed segment after EndFrame");

        var segments = sink.Snapshot().OfType<SpeakerSegmentFrame>().ToArray();
        var committed = Assert.Single(segments, s => s.IsFinal);
        Assert.Equal(0, committed.Speaker);

        // The first revision of a turn that starts the session has no previous speaker to lean on, so it is published
        // as unknown and only the closing read commits a label: the upsert stream is what upgrades it.
        var first = segments.OrderBy(s => s.Revision).First(s => s.SegmentId == committed.SegmentId);
        Assert.Equal(SpeakerSegmentUpdate.UnknownSpeaker, first.Speaker);
        Assert.True(committed.Confidence > 0, "a one-voice tone is describable");

        // Revisions of one id are strictly increasing: consumers upsert on the highest one.
        foreach (var group in segments.GroupBy(s => s.SegmentId))
            Assert.Equal(Enumerable.Range(0, group.Count()), group.Select(s => s.Revision));
    }

    [Fact]
    public async Task A_stalled_worker_becomes_a_timeline_gap_not_backpressure()
    {
        var segmentation = new TonePresenceSegmentation { Gate = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously) };
        var options = Options(queueCapacity: 1);

        await using var processor = new SpeakerDiarizationProcessor(segmentation, new ConstantEmbedding(), options);
        await using var sink = new Collector();
        processor.Link(sink);
        processor.Start();
        sink.Start();

        await BeginAsync(processor);

        // One 200 ms frame is exactly one model window, so the first push puts the worker inside inference.
        await PushAsync(processor, MonoTone.Pcm(0.2), chunkSamples: MonoTone.WindowSamples);
        await WaitUntilAsync(() => segmentation.Analyses > 0, "the worker should be inside inference");

        // While it is stuck, the data loop keeps accepting audio: one chunk queues, the rest become a gap.
        for (int i = 0; i < 5; i++)
            await processor.QueueFrameAsync(Audio(MonoTone.Pcm(0.2)));  // 5 chunks through a depth-1 queue

        await WaitUntilAsync(() => processor.SamplesSeen == 6 * MonoTone.WindowSamples, "the data loop accepted all six chunks");
        Assert.True(processor.DroppedSamples > 0, "a full queue must be reported as dropped audio");

        segmentation.Gate!.SetResult();
        await processor.QueueFrameAsync(new EndFrame());
        await WaitForAsync(sink, f => f.OfType<SpeakerSegmentFrame>().Any(s => s.IsFinal), "the tail is committed");

        // The gap is timeline, not silence attributed to nobody: the audio clock still counts what was lost.
        Assert.Equal(6 * (long)MonoTone.WindowSamples, processor.Diarizer.SamplesSeen);
    }

    [Fact]
    public async Task A_rate_mismatch_forwards_the_input_clock_but_diarizes_nothing()
    {
        await using var processor = new SpeakerDiarizationProcessor(
            new TonePresenceSegmentation(), new ConstantEmbedding(), Options());
        await using var sink = new Collector();
        processor.Link(sink);
        processor.Start();
        sink.Start();

        await BeginAsync(processor);
        await processor.QueueFrameAsync(Audio(MonoTone.Pcm(0.2), rate: 8_000));
        await processor.QueueFrameAsync(new EndFrame());
        await WaitForAsync(sink, f => f.OfType<EndFrame>().Any(), "EndFrame reaches the sink");

        var frames = sink.Snapshot();
        Assert.Empty(frames.OfType<SpeakerSegmentFrame>());
        Assert.Equal(0L, Assert.Single(frames.OfType<AudioRawFrame>()).StartSample);
        Assert.Equal(0, processor.Diarizer.SamplesSeen);
    }

    [Fact]
    public async Task Interleaved_frames_pass_through_untouched()
    {
        await using var processor = new SpeakerDiarizationProcessor(
            new TonePresenceSegmentation(), new ConstantEmbedding(), Options());
        await using var sink = new Collector();
        processor.Link(sink);
        processor.Start();
        sink.Start();

        await processor.QueueFrameAsync(new StartFrame());
        await processor.QueueFrameAsync(new TextFrame("hello"));
        await WaitForAsync(sink, f => f.OfType<TextFrame>().Any(), "a non-audio frame is forwarded");

        Assert.Single(sink.Snapshot().OfType<TextFrame>());
        Assert.Single(sink.Snapshot().OfType<StartFrame>());
    }
}
