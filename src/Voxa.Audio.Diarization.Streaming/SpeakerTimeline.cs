using Voxa.Audio.Diarization;
using Voxa.Frames;

namespace Voxa.Audio.Diarization.Streaming;

/// <summary>
/// The attribution stage's view of "who spoke when" (VLS-011), keyed by segment id so a revision replaces the state
/// it supersedes instead of appending a duplicate. Bounded: the oldest segments are dropped once
/// <see cref="SpeakerAttributionOptions.MaxTrackedSegments"/> is reached — lookups are always for the recent past.
/// </summary>
internal sealed class SpeakerTimeline
{
    private readonly List<Entry> _entries = [];
    private readonly Dictionary<string, int> _indexById = [];
    private readonly SpeakerAttributionOptions _options;

    public SpeakerTimeline(SpeakerAttributionOptions options)
        => _options = options ?? throw new ArgumentNullException(nameof(options));

    /// <summary>Number of segments currently tracked (test seam for the bound).</summary>
    public int Count => _entries.Count;

    /// <summary>Apply a segment upsert. An older revision arriving late is ignored (the stream is ordered, but a
    /// revision that lost a race must not roll a label back).</summary>
    public void Upsert(SpeakerSegmentFrame segment)
    {
        if (_indexById.TryGetValue(segment.SegmentId, out int index))
        {
            var existing = _entries[index];
            if (segment.Revision < existing.Revision) return;
            _entries[index] = new Entry(segment.SegmentId, segment.Speaker, segment.Start, segment.End, segment.Revision);
            return;
        }

        _entries.Add(new Entry(segment.SegmentId, segment.Speaker, segment.Start, segment.End, segment.Revision));
        _indexById[segment.SegmentId] = _entries.Count - 1;
        Trim();
    }

    /// <summary>Rewrite every tracked segment of a retired speaker to the surviving id (ids are never reused).</summary>
    public void Merge(int retired, int surviving)
    {
        for (int i = 0; i < _entries.Count; i++)
            if (_entries[i].Speaker == retired) _entries[i] = _entries[i] with { Speaker = surviving };
    }

    /// <summary>
    /// Attribute a span to a speaker: the greatest total overlap wins, and a span the diarizer has not covered yet
    /// borrows the speaker whose segment ends closest before it (within the continuity window). Returns
    /// <see cref="SpeakerSegmentUpdate.UnknownSpeaker"/> when there is nothing to say.
    /// </summary>
    /// <remarks>
    /// Overlap is summed per speaker, so one speaker's turn split across several segments (a revision, a merged
    /// neighbour) is not penalised against a single competing segment. Ties go to the earlier-starting segment, which
    /// keeps a text that straddles a speaker change attributed to whoever started first — deterministic, and the
    /// documented rule for "the text covers two speakers".
    /// </remarks>
    public int Attribute(double start, double end)
    {
        if (end < start) (start, end) = (end, start);

        Dictionary<int, double>? overlap = null;
        double bestOverlap = 0;
        int bestSpeaker = SpeakerSegmentUpdate.UnknownSpeaker;
        double bestStart = double.MaxValue;

        foreach (var entry in _entries)
        {
            double shared = Math.Min(end, entry.End) - Math.Max(start, entry.Start);
            if (shared <= 0) continue;

            overlap ??= [];
            overlap.TryGetValue(entry.Speaker, out double total);
            overlap[entry.Speaker] = total + shared;
        }

        if (overlap is not null)
        {
            foreach (var entry in _entries)
            {
                double shared = Math.Min(end, entry.End) - Math.Max(start, entry.Start);
                if (shared <= 0) continue;
                double total = overlap[entry.Speaker];
                if (total > bestOverlap + 1e-9 || (Math.Abs(total - bestOverlap) <= 1e-9 && entry.Start < bestStart))
                {
                    bestOverlap = total;
                    bestSpeaker = entry.Speaker;
                    bestStart = entry.Start;
                }
            }
            if (bestSpeaker != SpeakerSegmentUpdate.UnknownSpeaker) return bestSpeaker;
        }

        // Nothing covers the span yet: the diarizer commits behind the audio, so a transcript that arrives first
        // borrows the speaker who was talking immediately before it.
        int previous = SpeakerSegmentUpdate.UnknownSpeaker;
        double previousEnd = double.MinValue;
        foreach (var entry in _entries)
        {
            if (entry.End > start + 1e-9 || entry.End < previousEnd) continue;
            previousEnd = entry.End;
            previous = entry.Speaker;
        }

        if (previous == SpeakerSegmentUpdate.UnknownSpeaker) return SpeakerSegmentUpdate.UnknownSpeaker;
        return start - previousEnd <= _options.ContinuityWindow ? previous : SpeakerSegmentUpdate.UnknownSpeaker;
    }

    private void Trim()
    {
        int excess = _entries.Count - Math.Max(1, _options.MaxTrackedSegments);
        if (excess <= 0) return;

        _entries.RemoveRange(0, excess);
        _indexById.Clear();
        for (int i = 0; i < _entries.Count; i++) _indexById[_entries[i].Id] = i;
    }

    private readonly record struct Entry(string Id, int Speaker, double Start, double End, int Revision);
}
