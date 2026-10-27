using Microsoft.ML.OnnxRuntime;
using Voxa.Audio.Onnx;

namespace Voxa.Audio.Diarization.Onnx;

/// <summary>
/// The streaming (frame-level) reading of pyannote segmentation-3.0 (VLS-011): one window in, per-frame
/// <b>per-speaker</b> probabilities out, overlap intact — the <see cref="ISpeakerMaskSegmentation"/> the online
/// diarizer drives. The batch engine (<see cref="PyannoteOnnxSegmentation"/>) shares the same graph and the same
/// powerset layout, but collapses each frame to speech/silence because a finished recording only needs regions;
/// this one keeps the mask, which is what speaker changes, overlapping speech and embedding spans are derived from.
///
/// <para>Geometry comes from the model's metadata (<see cref="PyannoteSegmentationModel"/>), so a re-export carries
/// its own parameters. <c>windowSamples</c> may shorten the window to trade accuracy for lookahead — but only if the
/// graph actually accepts a dynamic length; a fixed-length export refuses at construction rather than silently
/// zero-padding or truncating.</para>
///
/// <para>The weights load once on the process-wide <see cref="OnnxModelHost"/>; this engine does not own the
/// session. One instance is single-threaded (it reuses an input buffer), which matches the diarizer's single worker.
/// </para>
/// </summary>
public sealed class PyannoteOnnxMaskSegmentation : ISpeakerMaskSegmentation
{
    private readonly IOnnxSession _session;
    private readonly PyannoteSegmentationModel _model;
    private readonly float[][] _powersetMapping;
    private readonly float[] _input;
    private readonly long[] _inputShape;

    /// <param name="modelPath">Resolved path to the pyannote segmentation ONNX (see <see cref="PyannoteSegmentationCatalog"/>).</param>
    /// <param name="host">The shared ONNX session host.</param>
    /// <param name="device">Execution device (CPU default).</param>
    /// <param name="windowSamples">
    /// Window length to run, or <c>0</c> for the model's own. Shorter windows cut the diarizer's lookahead
    /// (<c>(window − hop) / 2</c>) at some accuracy cost; longer ones are rejected by the graph.
    /// </param>
    public PyannoteOnnxMaskSegmentation(
        string modelPath, OnnxModelHost host, OnnxDevice device = OnnxDevice.Cpu, int windowSamples = 0)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelPath);
        ArgumentNullException.ThrowIfNull(host);

        _session = host.Load(modelPath, device);
        _model = PyannoteSegmentationModel.FromMetadata(_session.Session.ModelMetadata.CustomMetadataMap);
        _powersetMapping = PowersetSegmentationDecoder.BuildPowersetMapping(
            _model.NumClasses, _model.NumSpeakers, _model.PowersetMaxClasses);

        WindowSamples = windowSamples > 0 ? windowSamples : _model.WindowSize;
        if (WindowSamples % _model.ReceptiveFieldShift != 0)
            throw new ArgumentException(
                $"Window {WindowSamples} must be a multiple of the model's receptive-field shift " +
                $"{_model.ReceptiveFieldShift}.", nameof(windowSamples));

        var declared = _session.Session.InputMetadata[_session.InputNames[0]].Dimensions;
        int declaredLength = declared.Length > 0 ? declared[^1] : -1;
        if (declaredLength > 0 && declaredLength != WindowSamples)
            throw new ArgumentException(
                $"This export has a fixed {declaredLength}-sample input; windowSamples={WindowSamples} would be " +
                "rejected by the graph. Re-export with a dynamic length axis to shorten the window.", nameof(windowSamples));

        _input = new float[WindowSamples];
        _inputShape = [1, 1, WindowSamples];
    }

    /// <inheritdoc />
    public int SampleRate => _model.SampleRate;

    /// <inheritdoc />
    public int WindowSamples { get; }

    /// <inheritdoc />
    public int FrameShiftSamples => _model.ReceptiveFieldShift;

    /// <inheritdoc />
    public int FrameOffsetSamples => _model.ReceptiveFieldSize / 2;

    /// <inheritdoc />
    public int MaxLocalSpeakers => _model.NumSpeakers;

    /// <inheritdoc />
    public SpeakerMaskWindow Analyze(ReadOnlySpan<float> window)
    {
        if (window.Length != WindowSamples)
            throw new ArgumentException(
                $"Expected exactly {WindowSamples} samples; got {window.Length}.", nameof(window));

        var probabilities = new float[MaxLocalSpeakers];
        int classes = _model.NumClasses;
        string inputName = _session.InputNames[0];
        string outputName = _session.OutputNames[0];

        window.CopyTo(_input);
        using var input = OrtValue.CreateTensorValueFromMemory(_input, _inputShape);
        var inputs = new Dictionary<string, OrtValue> { [inputName] = input };
        using var runOptions = new RunOptions();
        using var results = _session.Session.Run(runOptions, inputs, [outputName]);

        var data = results[0].GetTensorDataAsSpan<float>();
        if (classes <= 0 || data.Length % classes != 0)
            throw new InvalidOperationException(
                $"Pyannote output length {data.Length} is not a multiple of {classes} classes — the model's " +
                "num_classes metadata disagrees with its output width.");

        int frames = data.Length / classes;
        var rows = new float[frames][];
        for (int f = 0; f < frames; f++)
        {
            var row = new float[MaxLocalSpeakers];
            PowersetSegmentationDecoder.FrameSpeakerProbabilities(data.Slice(f * classes, classes), _powersetMapping, row);
            rows[f] = row;
        }

        return new SpeakerMaskWindow(rows);
    }
}
