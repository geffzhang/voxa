# Voxa.Audio.Diarization

Speaker **diarization** for Voxa (VLS-005) — *"who spoke when"*, filling the long-present but always-null
`TranscriptionFrame.SpeakerId` for batch / meeting transcription.

This package ships the **seams + a pure-C# pipeline** — not a model:

- **`ISpeakerSegmentation`** — audio → speech regions (model-backed; e.g. Pyannote).
- **`ISpeakerEmbedding`** — a speech span → a fixed-width speaker vector (model-backed; e.g. WeSpeaker).
- **`IDiarizer`** / **`DiarizationPipeline`** — the orchestrator. The pipeline composes a segmenter + an
  embedder and does the rest in **pure C#**: form regions → embed each → **constrained agglomerative
  clustering by cosine distance (centroid linkage)** → stable speaker ids, with consecutive same-speaker
  regions merged into one turn. Centroid linkage matches what speech-core / pyannote calibrate the `0.715`
  default threshold against.
- **`DiarizerConfig`** — tunables, defaulted to speech-core's values. The one that matters is
  `ClusteringThreshold` (cosine-distance merge ceiling, default `0.715`); `MinSpeakers` / `MaxSpeakers`
  (`0` = auto) force a floor / cap.

## Why the orchestration is dependency-free

The pipeline references **no ML runtime** (and not even `Voxa.Core`) — steps 3–4 take `float[]` embeddings and
emit `DiarizedSegment[]` with no I/O. That is what makes the clustering testable on **hand-built synthetic
embeddings** in the default lane (two tight groups → two speakers, threshold sensitivity, speaker-count caps,
determinism) with no model download, mirroring speech-core's runtime-free `DiarizationPipeline`.

## Streaming / online diarization (VLS-011)

The pipeline above is a **batch** diarizer: global clustering, re-reads the recording, nothing final until the
audio ends. For a live call the same package ships the online counterpart — see
[VLS-011](docs/specifications/vls-011-streaming-diarization-spec.html):

- **`ISpeakerMaskSegmentation`** + **`SpeakerMaskWindow`** — the frame-level seam that keeps the speaker
  information: one probability per *local* speaker per 10 ms frame, so a hand-over is two channels, an overlap is
  two non-zero entries on one frame, and each channel gives a span worth embedding. (The older
  `ISpeakerSegmentation` returns speech regions — the collapse this seam exists to avoid.)
- **`StreamingDiarizer`** — `Push(audio)` / `NotifyGap(dropped)` / `Flush()`. Holds three kinds of state and
  nothing else: a fixed-capacity `AudioRingBuffer` addressed by absolute sample index, the window schedule with
  its commit frontier (`CommitFrontierSeconds` = the lookahead this capability pays), and the session's
  `SpeakerPrototypes`.
- **`StreamingDiarizerConfig`** — window/hop (default: the engine's, 10% hop), onset/offset hysteresis, evidence
  floors, match/new-speaker thresholds, continuity window, speaker cap, merge bounds, revision window, retained
  seconds.
- **`SpeakerPrototypes`** — stable ids in creation order, never reused; a short or low-confidence turn *borrows*
  the previous speaker (or answers `UnknownSpeaker`) instead of inventing one; only a closing read enrols a voice.
- **`SpeakerSegmentUpdate`** — the upsert protocol: keyed by `Id`, `Revision` monotonic from 0, `IsFinal` = the
  geometry is settled, `Confidence` = how much the *label* deserves trust.

Frame-level plumbing (the clock stamp, the off-loop worker, the transcript join) lives in
**`Voxa.Audio.Diarization.Streaming`**; the model-backed segmentation seam lives in
**`Voxa.Audio.Diarization.Onnx`** (`PyannoteOnnxMaskSegmentation`). Like the batch pipeline, everything here is
runtime-free and testable against two tones — no model, no download.

## Not yet here (needs real pinned models / a consumer)

- The **reference ONNX implementations** (Pyannote segmentation + WeSpeaker embedding) live in a separate
  opt-in `Voxa.Audio.Diarization.Onnx` package — deferred until their models are pinned (real SHA-256 +
  cleared licences) and built on the VLS-006 ONNX host.
- The **`voxa transcribe --diarize`** CLI verb that writes `SpeakerId` onto the transcript — the natural first
  consumer, which only does something once the ONNX impls exist.
- Speaker **identification** (enrollment against known voices) is out of scope for both pipelines (follow-ups once
  embeddings exist). Streaming ids are per-session clusters, not a directory.
