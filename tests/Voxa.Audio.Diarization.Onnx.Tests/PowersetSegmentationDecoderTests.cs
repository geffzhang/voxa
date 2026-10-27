using Voxa.Audio.Diarization;

namespace Voxa.Audio.Diarization.Onnx.Tests;

/// <summary>
/// Unit tests for the pure powerset→regions decoder — the validatable core (no model, no network), pinned
/// against the sherpa-onnx reference logic.
/// </summary>
public class PowersetSegmentationDecoderTests
{
    [Fact]
    public void Powerset_mapping_matches_pyannote_layout()
    {
        // seg-3.0: 3 speakers, up to 2 concurrent → 7 classes: silence, 3 singles, 3 pairs (in that order).
        var m = PowersetSegmentationDecoder.BuildPowersetMapping(numClasses: 7, numSpeakers: 3, powersetMaxClasses: 2);

        Assert.Equal([0f, 0f, 0f], m[0]); // silence
        Assert.Equal([1f, 0f, 0f], m[1]);
        Assert.Equal([0f, 1f, 0f], m[2]);
        Assert.Equal([0f, 0f, 1f], m[3]);
        Assert.Equal([1f, 1f, 0f], m[4]); // {0,1}
        Assert.Equal([1f, 0f, 1f], m[5]); // {0,2}
        Assert.Equal([0f, 1f, 1f], m[6]); // {1,2}
    }

    [Fact]
    public void Speech_classes_are_everything_but_silence()
    {
        var mapping = PowersetSegmentationDecoder.BuildPowersetMapping(7, 3, 2);
        Assert.Equal([false, true, true, true, true, true, true], PowersetSegmentationDecoder.SpeechClasses(mapping));
    }

    [Fact]
    public void Frame_activity_is_argmax_then_speech_lookup()
    {
        bool[] speech = [false, true, true]; // class 0 silence, 1/2 speech
        Assert.Equal(0f, PowersetSegmentationDecoder.FrameActivity([0.9f, 0.1f, 0.0f], speech)); // argmax 0 → silence
        Assert.Equal(1f, PowersetSegmentationDecoder.FrameActivity([0.1f, 0.8f, 0.1f], speech)); // argmax 1 → speech
        Assert.Equal(1f, PowersetSegmentationDecoder.FrameActivity([0.1f, 0.2f, 0.7f], speech)); // argmax 2 → speech
    }

    [Fact]
    public void Powerset_max_classes_above_two_is_rejected()
        => Assert.Throws<NotSupportedException>(() => PowersetSegmentationDecoder.BuildPowersetMapping(15, 4, 3));

    private static PyannoteSegmentationModel TestModel(int receptiveFieldSize = 0) => new(
        SampleRate: 16000, WindowSize: 1000, WindowShift: 100,
        ReceptiveFieldSize: receptiveFieldSize, ReceptiveFieldShift: 100,
        NumSpeakers: 3, PowersetMaxClasses: 2, NumClasses: 7);

    [Fact]
    public void Form_regions_single_chunk_two_runs()
    {
        // One full window (no padding); two speech runs → two regions. With a single chunk the Hamming-weighted
        // overlap-add collapses to the raw activity, so the onset/offset boundaries are exact.
        var activity = new[] { new float[] { 0, 1, 1, 0, 0, 1, 1, 0 } };
        var regions = PowersetSegmentationDecoder.FormRegions(activity, TestModel(), totalSamples: 1000, hasLastChunk: false);

        // scale = receptive_field_shift / sample_rate = 100/16000 = 0.00625; offset = 0 here.
        Assert.Equal(2, regions.Count);
        Assert.Equal(0.00625, regions[0].Start, 5);
        Assert.Equal(0.01875, regions[0].End, 5);
        Assert.Equal(0.03125, regions[1].Start, 5);
        Assert.Equal(0.04375, regions[1].End, 5);
    }

    [Fact]
    public void Form_regions_centres_frames_by_half_receptive_field()
    {
        // receptive_field_size 320 → scale_offset = 320/16000*0.5 = 0.01 added to every boundary.
        var activity = new[] { new float[] { 0, 1, 1, 0 } };
        var regions = PowersetSegmentationDecoder.FormRegions(activity, TestModel(receptiveFieldSize: 320), 1000, false);
        Assert.Single(regions);
        Assert.Equal(1 * 0.00625 + 0.01, regions[0].Start, 5);
        Assert.Equal(3 * 0.00625 + 0.01, regions[0].End, 5);
    }

    [Fact]
    public void Form_regions_stitches_two_overlapping_chunks()
    {
        // Two windows overlapping by 200 samples (windowShift 200, windowSize 400, receptive_field_shift 100):
        //   chunk 0 starts at global frame 0, chunk 1 at frame 2 (int(1*200/100 + 0.5)).
        // numFramesGlobal = (400 + 1*200)/100 + 1 = 7; Hamming(4) = [0.08, 0.77, 0.77, 0.08].
        var model = new PyannoteSegmentationModel(
            SampleRate: 16000, WindowSize: 400, WindowShift: 200,
            ReceptiveFieldSize: 0, ReceptiveFieldShift: 100,
            NumSpeakers: 3, PowersetMaxClasses: 2, NumClasses: 7);
        var chunks = new[]
        {
            new float[] { 0, 1, 1, 0 }, // frames 0..3
            new float[] { 1, 1, 0, 0 }, // frames 2..5
        };

        var regions = PowersetSegmentationDecoder.FormRegions(chunks, model, totalSamples: 600, hasLastChunk: false);

        // Stitched classification ≈ [0, 1, 1, 0.906, 0, 0, 0] (frame 3 = 0.77/0.85 from the overlap, > 0.5 → still
        // speech), so a single region [frame 1 .. frame 4) = [0.00625, 0.025) at scale 100/16000.
        Assert.Single(regions);
        Assert.Equal(0.00625, regions[0].Start, 5);
        Assert.Equal(0.025, regions[0].End, 5);
    }

    [Fact]
    public void Form_regions_all_silence_and_all_speech()
    {
        Assert.Empty(PowersetSegmentationDecoder.FormRegions([new float[] { 0, 0, 0, 0 }], TestModel(), 1000, false));

        var all = PowersetSegmentationDecoder.FormRegions([new float[] { 1, 1, 1, 1 }], TestModel(), 1000, false);
        Assert.Single(all);                       // one region spanning the speech, closed at the recording end
        Assert.Equal(0.0, all[0].Start, 5);
    }

    [Fact]
    public void Form_regions_no_chunks_is_empty()
        => Assert.Empty(PowersetSegmentationDecoder.FormRegions([], TestModel(), 0, false));

    [Fact]
    public void Hamming_window_matches_numpy()
    {
        Assert.Equal([1.0], PowersetSegmentationDecoder.Hamming(1));
        var w = PowersetSegmentationDecoder.Hamming(8);
        Assert.Equal(0.08, w[0], 5);   // endpoints
        Assert.Equal(0.08, w[7], 5);
        Assert.Equal(w[0], w[7], 9);   // symmetric
        Assert.Equal(w[1], w[6], 9);
        Assert.True(w[3] > 0.9 && w[4] > 0.9); // peak near the centre
    }

    // ---------------------------------------------------------------- speaker-level decode (VLS-011)

    /// <summary>seg-3.0's head: 3 local speakers, up to 2 concurrent, 7 powerset classes.</summary>
    private static float[][] Seg3Mapping()
        => PowersetSegmentationDecoder.BuildPowersetMapping(numClasses: 7, numSpeakers: 3, powersetMaxClasses: 2);

    [Fact]
    public void A_silence_frame_marginalises_to_zero_for_every_speaker()
    {
        var probabilities = new float[3];
        PowersetSegmentationDecoder.FrameSpeakerProbabilities([40f, -40f, -40f, -40f, -40f, -40f, -40f], Seg3Mapping(), probabilities);

        Assert.All(probabilities, p => Assert.Equal(0f, p, 6));
    }

    [Fact]
    public void A_single_speaker_frame_puts_all_the_mass_on_that_speaker()
    {
        var probabilities = new float[3];
        PowersetSegmentationDecoder.FrameSpeakerProbabilities([-40f, -40f, 40f, -40f, -40f, -40f, -40f], Seg3Mapping(), probabilities);

        Assert.Equal(0f, probabilities[0], 5);
        Assert.Equal(1f, probabilities[1], 5);
        Assert.Equal(0f, probabilities[2], 5);
    }

    [Fact]
    public void An_overlapping_frame_keeps_both_speakers_and_their_probabilities()
    {
        // Class 4 is the {0,1} pair: both speakers get the class's whole probability, so two non-zero entries on one
        // frame *are* the overlap — the information a speech/silence collapse would have thrown away.
        var probabilities = new float[3];
        PowersetSegmentationDecoder.FrameSpeakerProbabilities([-40f, -40f, -40f, -40f, 40f, -40f, -40f], Seg3Mapping(), probabilities);

        Assert.Equal(1f, probabilities[0], 5);
        Assert.Equal(1f, probabilities[1], 5);
        Assert.Equal(0f, probabilities[2], 5);
    }

    [Fact]
    public void Two_competing_classes_split_the_mass_between_their_speakers()
    {
        // Equal logits for {0} and {1,2}: softmax gives 0.5 each, so speaker 0 gets 0.5 and speakers 1 and 2 get 0.5.
        var probabilities = new float[3];
        PowersetSegmentationDecoder.FrameSpeakerProbabilities([-40f, 0f, -40f, -40f, -40f, -40f, 0f], Seg3Mapping(), probabilities);

        Assert.Equal(0.5f, probabilities[0], 5);   // 0.5 from class 1
        Assert.Equal(0.5f, probabilities[1], 5);   // 0.5 from class 6
        Assert.Equal(0.5f, probabilities[2], 5);   // 0.5 from class 6
        Assert.Equal(1.5f, probabilities[0] + probabilities[1] + probabilities[2], 5);
    }

    [Fact]
    public void The_marginal_is_a_softmax_not_a_normalised_marginal()
    {
        // Per-frame speaker probabilities deliberately do not sum to 1: a pair class contributes to both of its
        // speakers, and the diarizer's onset/offset logic reads them as independent tracks.
        var probabilities = new float[3];
        PowersetSegmentationDecoder.FrameSpeakerProbabilities([-40f, -40f, -40f, -40f, 40f, -40f, -40f], Seg3Mapping(), probabilities);

        float pair = probabilities[0];
        Assert.True(probabilities[1] > 0.9 * pair, "both members of the pair share one class's probability");
        Assert.True(probabilities[0] + probabilities[1] + probabilities[2] > 1.5, "the pair is counted twice");
    }

    [Fact]
    public void A_width_mismatch_is_rejected()
    {
        var mapping = Seg3Mapping();
        Assert.Throws<ArgumentException>(() =>
            PowersetSegmentationDecoder.FrameSpeakerProbabilities([0f, 0f, 0f], mapping, new float[3]));
        Assert.Throws<ArgumentException>(() =>
            PowersetSegmentationDecoder.FrameSpeakerProbabilities(new float[7], mapping, new float[2]));
    }

    [Fact]
    public void A_reused_destination_is_cleared_first()
    {
        var probabilities = new float[3];
        Array.Fill(probabilities, 0.5f);
        PowersetSegmentationDecoder.FrameSpeakerProbabilities([40f, -40f, -40f, -40f, -40f, -40f, -40f], Seg3Mapping(), probabilities);

        Assert.All(probabilities, p => Assert.Equal(0f, p, 6));
    }
}
