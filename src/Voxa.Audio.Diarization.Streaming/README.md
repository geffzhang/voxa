# Voxa.Audio.Diarization.Streaming

Frame-level plumbing for streaming speaker diarization (VLS-011). The state machine lives in
`Voxa.Audio.Diarization` (`StreamingDiarizer`) and the model-backed seam in `Voxa.Audio.Diarization.Onnx`; this
package is the part that touches a pipeline.

## `SpeakerDiarizationProcessor`

Placed on the **raw input, before the VAD** — diarization needs the continuous stream, silence included, and the
ungated audio is the only place it exists. It does two jobs:

1. **Owns the input clock.** Every `AudioRawFrame` it forwards is stamped with its absolute start sample
   (`AudioRawFrame.StartSample`), so the STT's utterance span (`TranscriptionFrame.StartTime/EndTime`) and the
   diarizer's segments are the *same* clock. A chunk replayed by the VAD's pre-roll keeps its stamp.
2. **Feeds the diarizer off the data path.** PCM16 → float → a bounded queue (`TryWrite` only), consumed by one
   worker task that calls `Push` / `NotifyGap` / `Flush`. Model inference never runs on the data loop: a slow
   window delays diarization and, once the queue is full, becomes a **timeline gap** — never backpressure on the
   conversation.

Output is a stream of `SpeakerSegmentFrame` upserts (keyed by `SegmentId`, higher `Revision` replaces) plus
`SpeakerMergedFrame` notices, riding the normal data channel so they cannot overtake the audio they describe.

```csharp
var processor = new SpeakerDiarizationProcessor(segmentation, embedding, new SpeakerDiarizationOptions());
// ... link into the pipeline; wires up nothing unless Voxa:Diarization:Engine = "Streaming"
```

Teardown (`EndFrame`) completes the queue, waits `FlushTimeout` for the worker, reports any remaining drops as a
gap, then flushes: the tail windows run and every open turn is committed.

A sample-rate or channel-count mismatch is refused once, loudly, and the session forwards audio on the clock
without diarizing it — a wrong-rate embedding is worse than no embedding.

## `SpeakerAttributionProcessor`

Downstream of STT (after the transcription filter). Consumes `SpeakerSegmentFrame` / `SpeakerMergedFrame` from the
diarizer and the timed `TranscriptionFrame`s from STT, and fills `TranscriptionFrame.SpeakerId` by an interval join
on the shared clock:

- **greatest total overlap** wins, summed per speaker (a turn split across segments is not penalised);
- **ties go to the earlier-starting segment** — the documented answer for text that crosses a speaker change;
- a span the diarizer has not committed yet (it lags by the model's lookahead) **borrows** the speaker who was
  talking immediately before it, inside `ContinuityWindow`.

It never blocks, never waits for the diarizer, and never overwrites a `SpeakerId` an STT engine already set. The
segment stream stays authoritative: a later revision carries the corrected label on the same clock, so a consumer
can re-join history at any time. `SpeakerTimeline` (internal) is the bounded lookup, and it is what the tests drive
directly.
