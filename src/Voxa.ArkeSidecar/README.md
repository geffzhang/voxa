# Voxa Arke Sidecar

Loopback-only local speech service for Arke Studio. Publish one complete directory per Windows architecture:

```powershell
dotnet publish Voxa.ArkeSidecar.csproj -c Release -r win-x64 --self-contained true -o ../../publish/Voxa.ArkeSidecar-win-x64
dotnet publish Voxa.ArkeSidecar.csproj -c Release -r win-arm64 --self-contained true -o ../../publish/Voxa.ArkeSidecar-win-arm64
```

Launch arguments are discrete name/value pairs:

```text
Voxa.ArkeSidecar.exe
  --host 127.0.0.1
  --port 5901
  --kokoro-model <model.onnx>
  --kokoro-config <config.json>
  --kokoro-voices <directory containing the six .bin files>
  --whisper-model <ggml model .bin>
  --espeak <espeak-ng.exe>
  --espeak-data <espeak-ng-data directory or its parent>
```

`--host` defaults to `127.0.0.1` and rejects hostnames and non-loopback addresses. `--port` defaults to `PORT`, then `5901`. Missing speech assets do not prevent startup; `GET /health` reports a safe `unavailableReason` until all required paths exist.

The publish directory is the deployment unit. It contains `Voxa.ArkeSidecar.exe`, the self-contained .NET runtime, ONNX Runtime, WhisperCpp native libraries, and managed dependencies. Kokoro/Whisper models, Kokoro voices, and espeak-ng are external inputs and are not included.
