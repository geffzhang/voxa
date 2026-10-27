using Voxa.Audio.Diarization;
using Voxa.Frames;

namespace Voxa.Audio.Diarization.Streaming.Tests;

/// <summary>
/// The interval join behind transcript attribution (VLS-011): greatest overlap wins, ties go to the earlier speaker,
/// a span the diarizer has not committed yet borrows its predecessor, and revisions/merges rewrite state rather than
/// piling up.
/// </summary>
public class SpeakerTimelineTests
{
    private static SpeakerTimeline Timeline(double continuityWindow = 1.5, int maxTracked = 512)
        => new(new SpeakerAttributionOptions { ContinuityWindow = continuityWindow, MaxTrackedSegments = maxTracked });

    private static SpeakerSegmentFrame Segment(
        string id, int speaker, double start, double end, int revision = 0, bool final = false)
        => new(id, speaker, start, end, final, revision, Confidence: 1.0);

    [Fact]
    public void The_greatest_overlap_wins()
    {
        var timeline = Timeline();
        timeline.Upsert(Segment("a", 0, 0.0, 1.0));
        timeline.Upsert(Segment("b", 1, 1.0, 2.0));

        // 0.8–1.4 shares 0.2 s with the first turn and 0.4 s with the second: the text belongs to the second.
        Assert.Equal(1, timeline.Attribute(0.8, 1.4));
    }

    [Fact]
    public void A_tie_goes_to_the_speaker_who_started_first()
    {
        var timeline = Timeline();
        timeline.Upsert(Segment("a", 7, 0.0, 1.0));
        timeline.Upsert(Segment("b", 8, 1.0, 2.0));

        // Exactly half of the text on each side: whoever started talking first keeps it (deterministic rule).
        Assert.Equal(7, timeline.Attribute(0.5, 1.5));
    }

    [Fact]
    public void One_speaker_split_across_segments_is_not_penalised()
    {
        var timeline = Timeline();
        timeline.Upsert(Segment("a1", 3, 0.0, 0.5));
        timeline.Upsert(Segment("a2", 3, 0.5, 1.0));  // a later turn of the same voice
        timeline.Upsert(Segment("b", 4, 1.0, 2.0));

        // Overlap is summed per speaker: 0.75 s for speaker 3 against 0.25 s for speaker 4.
        Assert.Equal(3, timeline.Attribute(0.25, 1.25));
    }

    [Fact]
    public void A_span_nobody_has_committed_yet_borrows_its_predecessor_within_the_window()
    {
        var timeline = Timeline(continuityWindow: 0.5);
        timeline.Upsert(Segment("a", 2, 0.0, 1.0));

        // The diarizer lags by its lookahead: the transcript that arrives first still gets the speaker who just spoke.
        Assert.Equal(2, timeline.Attribute(1.2, 1.4));
        Assert.Equal(SpeakerSegmentUpdate.UnknownSpeaker, timeline.Attribute(1.8, 2.0));
    }

    [Fact]
    public void An_older_revision_arriving_late_does_not_roll_a_label_back()
    {
        var timeline = Timeline();
        timeline.Upsert(Segment("a", 5, 0.0, 1.0, revision: 3));
        timeline.Upsert(Segment("a", 6, 0.0, 1.0, revision: 1)); // stale

        Assert.Equal(5, timeline.Attribute(0.2, 0.4));
        Assert.Equal(1, timeline.Count);
    }

    [Fact]
    public void A_merge_rewrites_the_labels_it_held()
    {
        var timeline = Timeline();
        timeline.Upsert(Segment("a", 1, 0.0, 1.0));
        timeline.Upsert(Segment("b", 0, 1.0, 2.0));

        timeline.Merge(retired: 1, surviving: 0);

        Assert.Equal(0, timeline.Attribute(0.2, 0.4));
        Assert.Equal(0, timeline.Attribute(1.2, 1.4));
    }

    [Fact]
    public void The_timeline_is_bounded_and_keeps_the_newest_segments()
    {
        var timeline = Timeline(maxTracked: 4);
        for (int i = 0; i < 10; i++) timeline.Upsert(Segment($"s{i}", i, i, i + 1.0));

        Assert.Equal(4, timeline.Count);
        Assert.Equal(9, timeline.Attribute(9.2, 9.4));                       // recent history survives
        Assert.Equal(SpeakerSegmentUpdate.UnknownSpeaker, timeline.Attribute(0.2, 0.4)); // the oldest fell out
    }

    [Fact]
    public void An_empty_timeline_says_nothing()
        => Assert.Equal(SpeakerSegmentUpdate.UnknownSpeaker, Timeline().Attribute(0.0, 1.0));

    [Fact]
    public void A_reversed_span_is_joined_the_same_way()
    {
        var timeline = Timeline();
        timeline.Upsert(Segment("a", 0, 1.0, 2.0));
        Assert.Equal(0, timeline.Attribute(1.6, 1.2));
    }
}
