namespace Voxa.Audio.Diarization;

/// <summary>
/// The session's speaker identities (VLS-011) — the state that turns per-turn embeddings into stable speaker
/// numbers. One prototype per speaker (an L2-normalised centroid plus how much evidence built it); a turn is
/// matched by cosine distance and folded back into the matched prototype, so a voice heard across many windows
/// keeps one id without ever re-clustering the session.
///
/// <para><b>Id stability.</b> Ids are handed out in creation order and never renumbered or reused. The only way an
/// id disappears is a merge (two prototypes proven to be one voice), and even then the surviving id is the older
/// one — a consumer's history keeps its meaning.</para>
///
/// <para><b>Short and low-confidence turns.</b> A turn may create a speaker only if it carries
/// <see cref="StreamingDiarizerConfig.MinNewSpeakerEvidence"/> seconds of speech <i>and</i>
/// <see cref="StreamingDiarizerConfig.MinTurnConfidence"/> mean activity. Everything else is <i>adopted</i>: the
/// previous speaker if the turn follows it inside <see cref="StreamingDiarizerConfig.ContinuityWindow"/>, otherwise
/// the nearest prototype — and the resulting segment is published as provisional, because the evidence is too thin
/// to commit to it. The one case that never guesses is "no prototype is remotely close and the turn is too thin":
/// it becomes <see cref="SpeakerSegmentUpdate.UnknownSpeaker"/> rather than inventing a speaker.</para>
/// </summary>
internal sealed class SpeakerPrototypes
{
    private sealed class Prototype
    {
        public required int Id { get; init; }
        public required double[] Centroid { get; set; } // L2-normalised
        public double Weight { get; set; }              // speech seconds, capped (bounds drift)
        public double TotalEvidence { get; set; }       // speech seconds, uncapped (merge eligibility)
    }

    private readonly StreamingDiarizerConfig _config;
    private readonly List<Prototype> _prototypes = [];
    private int _nextId;

    public SpeakerPrototypes(StreamingDiarizerConfig config)
        => _config = config ?? throw new ArgumentNullException(nameof(config));

    /// <summary>Speaker ids in creation order.</summary>
    public IReadOnlyList<int> SpeakerIds
    {
        get
        {
            var ids = new int[_prototypes.Count];
            for (int i = 0; i < _prototypes.Count; i++) ids[i] = _prototypes[i].Id;
            return ids;
        }
    }

    /// <summary>
    /// Resolve a turn to a speaker, creating or updating a prototype as the evidence allows. Never throws for thin
    /// evidence — it answers <see cref="SpeakerSegmentUpdate.UnknownSpeaker"/> instead, so a noisy blip cannot
    /// invent an identity.
    /// </summary>
    /// <param name="embedding">The turn's speaker vector (need not be unit-norm).</param>
    /// <param name="durationSeconds">Speech seconds in the turn.</param>
    /// <param name="confidence">Mean per-frame activity of the turn's own speaker track, in [0, 1].</param>
    /// <param name="previousSpeaker">Speaker of the most recent committed segment, or -1.</param>
    /// <param name="gapSeconds">Silence between that segment's end and this turn's start (∞ when none).</param>
    /// <param name="enrol">
    /// <c>false</c> keeps the prototypes read-only for everything except creating a speaker: a match still returns an
    /// id but does not move that speaker's centroid. Used for the provisional label of a turn that is still growing —
    /// a fragment must not drag a settled voice toward itself (an overlap in progress would otherwise teach a
    /// speaker somebody else's voice). The turn's closing decision is what enrols.
    /// </param>
    public SpeakerDecision Decide(
        float[] embedding,
        double durationSeconds,
        double confidence,
        int previousSpeaker,
        double gapSeconds,
        bool enrol = true)
    {
        ArgumentNullException.ThrowIfNull(embedding);

        bool evidenced = durationSeconds >= _config.MinNewSpeakerEvidence
                         && confidence >= _config.MinTurnConfidence;

        if (_prototypes.Count == 0)
        {
            if (!evidenced)
                return new SpeakerDecision(SpeakerSegmentUpdate.UnknownSpeaker, double.MaxValue, Confident: false, Created: false);
            var created = Create(embedding, durationSeconds);
            return new SpeakerDecision(created, 0, Confident: true, Created: true);
        }

        var (best, distance) = Nearest(embedding);

        if (distance <= _config.MatchThreshold)
        {
            if (enrol) Update(best, embedding, durationSeconds);
            return new SpeakerDecision(best.Id, distance, Confident: true, Created: false);
        }

        if (evidenced && distance > _config.NewSpeakerThreshold && _prototypes.Count < _config.MaxSpeakers)
        {
            var created = Create(embedding, durationSeconds);
            return new SpeakerDecision(created, distance, Confident: true, Created: true);
        }

        // Thin evidence (short or low-confidence): prefer the speaker who was just talking. This is the
        // "mm-hm" / backchannel case — a 300 ms blip in a conversation is almost always the same person.
        if (!evidenced && previousSpeaker >= 0 && gapSeconds <= _config.ContinuityWindow && Find(previousSpeaker) is { } previous)
        {
            if (enrol) Update(previous, embedding, durationSeconds);
            return new SpeakerDecision(previous.Id, distance, Confident: false, Created: false);
        }

        // Dead band (evidenced but not separable) or an exhausted speaker cap: attach to the nearest prototype, with
        // a label confidence that says so.
        if (enrol) Update(best, embedding, durationSeconds);
        return new SpeakerDecision(best.Id, distance, Confident: false, Created: false);
    }

    /// <summary>
    /// Fold every prototype that has turned out to be the same voice into its older sibling. A speaker is only
    /// mergeable while its evidence is below <see cref="StreamingDiarizerConfig.MergeEvidence"/> — a voice that has
    /// carried a long stretch of the conversation is never retired into a newcomer. Survivors are the lower (older)
    /// id; the retired id is reported so consumers can remap history older than the revision window.
    /// </summary>
    public IReadOnlyList<SpeakerMergeInfo> Merge(double nowSeconds)
    {
        List<SpeakerMergeInfo>? merges = null;

        bool merged = true;
        while (merged && _prototypes.Count > 1)
        {
            merged = false;
            var (a, b) = ClosestPair();
            if (a is null || b is null) break;
            if (Distance(a.Centroid, b.Centroid) >= _config.MergeThreshold) break;

            // Retire the younger speaker; it must still be a newcomer to be retirable.
            var (retired, surviving) = a.Id > b.Id ? (a, b) : (b, a);
            if (retired.TotalEvidence >= _config.MergeEvidence) break;

            retired.TotalEvidence += surviving.TotalEvidence;
            surviving.Weight = Math.Min(_config.PrototypeWeightCap, surviving.Weight + retired.Weight);
            _prototypes.Remove(retired);
            (merges ??= []).Add(new SpeakerMergeInfo(retired.Id, surviving.Id, nowSeconds));
            merged = true;
        }

        return (IReadOnlyList<SpeakerMergeInfo>?)merges ?? [];
    }

    private Prototype? Find(int id)
    {
        foreach (var p in _prototypes) if (p.Id == id) return p;
        return null;
    }

    private int Create(float[] embedding, double durationSeconds)
    {
        var prototype = new Prototype
        {
            Id = _nextId++,
            Centroid = Normalize(embedding),
            Weight = Math.Min(durationSeconds, _config.PrototypeWeightCap),
            TotalEvidence = durationSeconds,
        };
        _prototypes.Add(prototype);
        return prototype.Id;
    }

    // Weighted centroid update that freezes once the prototype has seen PrototypeWeightCap seconds: the
    // "accepted" weight shrinks to zero, so a long session cannot drift a settled voice away from itself.
    private void Update(Prototype prototype, float[] embedding, double durationSeconds)
    {
        prototype.TotalEvidence += durationSeconds;
        double headroom = _config.PrototypeWeightCap - prototype.Weight;
        if (headroom <= 0 || durationSeconds <= 0) return;

        double add = Math.Min(durationSeconds, headroom);
        double total = prototype.Weight + add;
        var unit = Normalize(embedding);
        for (int i = 0; i < prototype.Centroid.Length; i++)
            prototype.Centroid[i] = (prototype.Centroid[i] * prototype.Weight + unit[i] * add) / total;

        NormalizeInPlace(prototype.Centroid);
        prototype.Weight = total;
    }

    private (Prototype Best, double Distance) Nearest(float[] embedding)
    {
        Prototype? best = null;
        double bestDistance = double.MaxValue;
        foreach (var candidate in _prototypes)
        {
            double distance = Distance(candidate.Centroid, embedding);
            if (distance < bestDistance)
            {
                bestDistance = distance;
                best = candidate;
            }
        }
        return (best!, bestDistance);
    }

    // Closest pair of prototypes; ties resolve to the earliest insertion order (deterministic merges).
    private (Prototype? A, Prototype? B) ClosestPair()
    {
        Prototype? bestA = null, bestB = null;
        double best = double.MaxValue;
        for (int a = 0; a < _prototypes.Count; a++)
        {
            for (int b = a + 1; b < _prototypes.Count; b++)
            {
                double distance = Distance(_prototypes[a].Centroid, _prototypes[b].Centroid);
                if (distance < best)
                {
                    best = distance;
                    bestA = _prototypes[a];
                    bestB = _prototypes[b];
                }
            }
        }
        return (bestA, bestB);
    }

    // Cosine distance over the prototype's double-precision centroid and a float embedding (the seam's vector
    // type). Same definition the batch clustering uses; zero-magnitude input is maximally dissimilar (2), never NaN.
    private static double Distance(double[] a, float[] b)
    {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += (double)b[i] * b[i];
        }
        if (normA <= 0 || normB <= 0) return 2.0;
        return 1.0 - Math.Clamp(dot / (Math.Sqrt(normA) * Math.Sqrt(normB)), -1.0, 1.0);
    }

    private static double Distance(double[] a, double[] b)
    {
        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        if (normA <= 0 || normB <= 0) return 2.0;
        return 1.0 - Math.Clamp(dot / (Math.Sqrt(normA) * Math.Sqrt(normB)), -1.0, 1.0);
    }

    private static double[] Normalize(float[] embedding)
    {
        var unit = new double[embedding.Length];
        for (int i = 0; i < embedding.Length; i++) unit[i] = embedding[i];
        NormalizeInPlace(unit);
        return unit;
    }

    // A zero vector has no direction; leave it zero (cosine distance to anything is 2, so it can never merge
    // silently — degenerate embeddings stay unmatched rather than collapsing speakers).
    private static void NormalizeInPlace(double[] vector)
    {
        double norm = 0;
        for (int i = 0; i < vector.Length; i++) norm += vector[i] * vector[i];
        if (norm <= 0) return;
        double scale = 1.0 / Math.Sqrt(norm);
        for (int i = 0; i < vector.Length; i++) vector[i] *= scale;
    }
}

/// <summary>Outcome of matching one turn against the session's speakers.</summary>
/// <param name="Speaker">Assigned speaker id, or <see cref="SpeakerSegmentUpdate.UnknownSpeaker"/>.</param>
/// <param name="Distance">Cosine distance to the best prototype (∞ when there was nothing to compare).</param>
/// <param name="Confident">True when the assignment rests on evidence; false for an adoption/nearest-attach.</param>
/// <param name="Created">True when this turn created the speaker it was assigned to.</param>
internal readonly record struct SpeakerDecision(int Speaker, double Distance, bool Confident, bool Created)
{
    /// <summary>
    /// How strongly the diarizer believes the label: 1.0 for a speaker this turn created, 1 − distance for a
    /// matched one, half that for a borrowed (adopted or nearest-attach) one, 0 for
    /// <see cref="SpeakerSegmentUpdate.UnknownSpeaker"/>.
    /// </summary>
    public double LabelConfidence
    {
        get
        {
            if (Speaker == SpeakerSegmentUpdate.UnknownSpeaker) return 0;
            if (Created) return 1.0;
            double similarity = Math.Clamp(1 - Distance, 0, 1);
            return Confident ? similarity : similarity * 0.5;
        }
    }
}
