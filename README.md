# Quick ASR Endpoint (Qwen3 ASR 0.6B)

A Dockerized FastAPI + Uvicorn service for speech-to-text with **Qwen3 ASR 0.6B**, including optional **vLLM backend** support:

- `POST /v1/transcribe` for file-based transcription (`.wav`, 16-bit PCM)
- `WS /v1/stream` for streaming transcription using PCM16LE chunks
- Bash scripts to build/run/stop the container

## 1) Build

```bash
./scripts/build.sh quick-asr-endpoint:latest
# optional: override torch build at image build time
# TORCH_SPEC="torch==2.4.1+cu118" TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118" ./scripts/build.sh quick-asr-endpoint:latest
# optional: pin a specific transformers version instead of source
# TRANSFORMERS_SPEC="transformers==4.57.1" ./scripts/build.sh quick-asr-endpoint:latest
```

### Build-time options

- `TORCH_SPEC` torch package spec used during image build (default: `torch==2.4.1`)
- `TORCH_INDEX_URL` optional pip index URL for torch wheels (for example `https://download.pytorch.org/whl/cu118`)
- `TRANSFORMERS_SPEC` transformers package spec used during image build (default: `git+https://github.com/huggingface/transformers.git`)
- `QWEN_ASR_SPEC` qwen-asr package spec used during image build (default: `qwen-asr[vllm]`)

## 2) Run

```bash
./scripts/run.sh quick-asr-endpoint:latest quick-asr-endpoint
# script waits until /health reports ready=true, fails with logs if startup crashes/times out
```

Optional environment variables:

- `PORT` (default: `8000`)
- `MODEL_ID` (default: `Qwen/Qwen3-ASR-0.6B`)
- `ASR_BACKEND` (`vllm` default; set `transformers` explicitly if GPU runtime is unavailable)
- `HF_TOKEN` for private/gated Hugging Face models
- `MODEL_REVISION` to pin a model revision/tag/commit
- `STARTUP_TIMEOUT` seconds to wait for `/health` (default: `300`)
- `POLL_INTERVAL` seconds between health checks (default: `2`)
- `ENABLE_GPU` (`auto` default; uses `--gpus all` when `nvidia-smi` is present. Set `1` to force GPU, `0` to disable.)
- `CHUNK_TIMEOUT_SECONDS` bailout timeout for per-transcription/chunk generation (default: `150` = 2m30s)

## 3) Health check

```bash
curl http://localhost:8000/health
```

## 4) Batch transcription

```bash
curl -X POST "http://localhost:8000/v1/transcribe" \
  -F "file=@sample.wav"
```

## 5) Streaming transcription (WebSocket)

### Protocol

1. Connect to `ws://localhost:8000/v1/stream`
2. Optionally send text command `sample_rate:16000`
3. Send binary PCM16LE audio chunks (mono, 16k recommended)
4. Send text command `end`
5. Receive `partial` events during streaming and `final` event at end

### Example Python client

```python
import asyncio
import websockets

async def run():
    uri = "ws://localhost:8000/v1/stream"
    async with websockets.connect(uri) as ws:
        print(await ws.recv())
        await ws.send("sample_rate:16000")

        with open("sample.pcm", "rb") as f:
            while chunk := f.read(3200):
                await ws.send(chunk)
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.05)
                    print(msg)
                except asyncio.TimeoutError:
                    pass

        await ws.send("end")
        while True:
            try:
                print(await ws.recv())
            except websockets.ConnectionClosed:
                break

asyncio.run(run())
```

## 6) Stop

```bash
./scripts/stop.sh quick-asr-endpoint
```

## Notes

- Streaming endpoint performs **incremental re-transcription** on accumulated audio for partial updates.
- The Docker image includes `git` so `TRANSFORMERS_SPEC` values like `git+https://...` install correctly.
- Build installs `qwen-asr` before `TRANSFORMERS_SPEC`, then reinstalls `TRANSFORMERS_SPEC` last to ensure custom Qwen3 architecture support is not downgraded by transitive deps.
- For GPU acceleration, run container with appropriate runtime (for example `--gpus all`) and CUDA-compatible base image.


## Startup/model loading behavior

- The API process now stays up even if model loading fails during startup.
- `GET /health` returns `status: degraded` and includes the model loading error when this happens.
- Default backend is `ASR_BACKEND=vllm` (recommended by Qwen3-ASR docs for throughput).
- You can set `ASR_BACKEND=auto` to try `vllm` first and automatically fall back to `transformers` when appropriate.
- If GPU runtime is unavailable, use `ASR_BACKEND=transformers` explicitly.
- `/health` also includes `active_backend` to show which backend was actually initialized.
- Transcription endpoints return `503` with error details until the model is configured correctly.
- Streaming and batch transcriptions bail out with timeout errors if generation exceeds `CHUNK_TIMEOUT_SECONDS` (default 2m30s).


## Troubleshooting

- Qwen3-ASR docs recommend the vLLM backend for best throughput; `scripts/run.sh` now defaults to `ASR_BACKEND=vllm` and can auto-enable `--gpus all` when supported.

- `scripts/run.sh` validates the image label (`org.opencontainers.image.title=quick-asr-endpoint`) to avoid accidentally running a different application image under a reused tag.

- If `/health` shows a model architecture error (for example unknown `qwen3_asr`), rebuild your image to pick up updated dependencies. The Transformers backend now attempts direct `pipeline(...)` loading first to support custom Qwen3 ASR config classes:

```bash
./scripts/build.sh quick-asr-endpoint:latest
# optional: override torch build at image build time
# TORCH_SPEC="torch==2.4.1+cu118" TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118" ./scripts/build.sh quick-asr-endpoint:latest
# optional: pin a specific transformers version instead of source
# TRANSFORMERS_SPEC="transformers==4.57.1" ./scripts/build.sh quick-asr-endpoint:latest
./scripts/run.sh quick-asr-endpoint:latest quick-asr-endpoint
```

- Ensure your upload path has no trailing space in the curl file argument, e.g. `-F "file=@../Quick-TTS-Endpoint/liz.wav"`.


## Use vLLM backend

The service defaults to `ASR_BACKEND=vllm` and installs `qwen-asr[vllm]` in Docker builds.

If needed, you can force Transformers backend at runtime:

```bash
ASR_BACKEND=transformers ./scripts/run.sh quick-asr-endpoint:latest quick-asr-endpoint
```

The `/health` response includes the active backend in both `ok` and `degraded` states.
