using Voxa.Frames;
using Voxa.Processors;

namespace Voxa.Audio.Diarization.Streaming.Tests;

/// <summary>
/// Transcript attribution on a real pipeline (VLS-011): the text gets the speaker whose segment covers most of its
/// span, a text that crosses a speaker change keeps a deterministic owner, and an engine that already labels speakers
/// (or a pipeline without timings) is left alone.
/// </summary>
public class SpeakerAttributionProcessorTests
{
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

    private static async Task<Frame[]> RunAsync(params Frame[] frames)
    {
        await using var processor = new SpeakerAttributionProcessor();
        await using var sink = new Collector();
        processor.Link(sink);
        processor.Start();
        sink.Start();

        foreach (var frame in frames) await processor.QueueFrameAsync(frame);

        // EndFrame goes through the stage, not around it: frames reach the sink from the stage's data loop, so the
        // terminator has to take the same path or it could overtake them.
        await processor.QueueFrameAsync(new EndFrame());

        var deadline = DateTime.UtcNow + TimeSpan.FromSeconds(5);
        while (DateTime.UtcNow < deadline)
        {
            var seen = sink.Snapshot();
            if (seen.OfType<EndFrame>().Any()) return seen;
            await Task.Delay(5);
        }
        throw new TimeoutException("the pipeline did not drain");
    }

    private static TranscriptionFrame Transcript(string text, double? start, double? end, string? speakerId = null)
        => new(text, IsFinal: true, Language: "en", SpeakerId: speakerId) { StartTime = start, EndTime = end };

    [Fact]
    public async Task A_timed_transcript_is_labelled_with_the_speaker_that_covers_it()
    {
        var frames = await RunAsync(
            new SpeakerSegmentFrame("seg-1", 2, 1.0, 2.0, IsFinal: true, Revision: 0, Confidence: 0.9),
            Transcript("hello there", 1.1, 1.6));

        var transcript = Assert.Single(frames.OfType<TranscriptionFrame>());
        Assert.Equal("speaker 2", transcript.SpeakerId);
        Assert.Equal("hello there", transcript.Text);
    }

    [Fact]
    public async Task The_segment_stream_survives_attribution()
    {
        var frames = await RunAsync(
            new SpeakerSegmentFrame("seg-1", 0, 0.0, 1.0, IsFinal: false, Revision: 4, Confidence: 0.5),
            Transcript("hi", 0.1, 0.4));

        // The segment is the authoritative record; attribution consumes it and passes it on.
        Assert.Single(frames.OfType<SpeakerSegmentFrame>());
        Assert.Equal("speaker 0", Assert.Single(frames.OfType<TranscriptionFrame>()).SpeakerId);
    }

    [Fact]
    public async Task An_untimed_transcript_is_forwarded_untouched()
    {
        var frames = await RunAsync(
            new SpeakerSegmentFrame("seg-1", 1, 0.0, 1.0, IsFinal: true, Revision: 0, Confidence: 1.0),
            Transcript("no span", null, null));

        Assert.Null(Assert.Single(frames.OfType<TranscriptionFrame>()).SpeakerId);
    }

    [Fact]
    public async Task A_speaker_the_stt_already_named_is_never_overwritten()
    {
        var frames = await RunAsync(
            new SpeakerSegmentFrame("seg-1", 1, 0.0, 1.0, IsFinal: true, Revision: 0, Confidence: 1.0),
            Transcript("from the engine", 0.1, 0.4, speakerId: "engine-speaker-X"));

        Assert.Equal("engine-speaker-X", Assert.Single(frames.OfType<TranscriptionFrame>()).SpeakerId);
    }

    [Fact]
    public async Task A_transcript_inside_a_merged_pair_gets_the_surviving_speaker()
    {
        var frames = await RunAsync(
            new SpeakerSegmentFrame("seg-1", 1, 0.0, 1.0, IsFinal: true, Revision: 0, Confidence: 1.0),
            new SpeakerMergedFrame(RetiredSpeaker: 1, SurvivingSpeaker: 0, AtSeconds: 1.0),
            Transcript("after the merge", 0.2, 0.6));

        Assert.Equal("speaker 0", Assert.Single(frames.OfType<TranscriptionFrame>()).SpeakerId);
        Assert.Single(frames.OfType<SpeakerMergedFrame>());
    }

    [Fact]
    public async Task A_transcript_with_no_covering_segment_borrows_the_previous_speaker()
    {
        var frames = await RunAsync(
            new SpeakerSegmentFrame("seg-1", 3, 0.0, 1.0, IsFinal: true, Revision: 0, Confidence: 1.0),
            Transcript("the diarizer is behind", 1.1, 1.3)); // still inside the continuity window

        Assert.Equal("speaker 3", Assert.Single(frames.OfType<TranscriptionFrame>()).SpeakerId);
    }
}
