import asyncio
import importlib
import importlib.util
import inspect
import io
import os
import wave
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-0.6B")
ASR_BACKEND = os.getenv("ASR_BACKEND", "vllm").lower()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_REVISION = os.getenv("MODEL_REVISION")
CHUNK_SAMPLE_RATE = 16000
CHUNK_WIDTH = 2  # int16
CHUNK_CHANNELS = 1
CHUNK_TIMEOUT_SECONDS = float(os.getenv("CHUNK_TIMEOUT_SECONDS", "150"))


class BaseASREngine:
    def transcribe_pcm16le(self, pcm_bytes: bytes, sample_rate: int = CHUNK_SAMPLE_RATE) -> str:
        raise NotImplementedError

    def transcribe_wav(self, wav_bytes: bytes) -> str:
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()

        if sample_width != CHUNK_WIDTH:
            raise ValueError("Only 16-bit PCM WAV is supported.")

        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)

        return self.transcribe_pcm16le(audio.tobytes(), sample_rate=sample_rate)


class TransformersASREngine(BaseASREngine):
    def __init__(self, model_id: str = MODEL_ID):
        # Prefer letting `pipeline(...)` resolve the right architecture when remote code
        # defines a custom config/model (e.g., qwen3_asr) instead of forcing Seq2Seq.
        try:
            self.pipe = pipeline(
                task="automatic-speech-recognition",
                model=model_id,
                token=HF_TOKEN,
                revision=MODEL_REVISION,
                trust_remote_code=True,
                torch_dtype=TORCH_DTYPE,
                device=0 if DEVICE == "cuda" else -1,
            )
            return
        except Exception:
            # Fallback for environments where direct pipeline loading fails.
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                model_id,
                torch_dtype=TORCH_DTYPE,
                low_cpu_mem_usage=True,
                use_safetensors=True,
                token=HF_TOKEN,
                revision=MODEL_REVISION,
                trust_remote_code=True,
            )
            model.to(DEVICE)
            processor = AutoProcessor.from_pretrained(
                model_id,
                token=HF_TOKEN,
                revision=MODEL_REVISION,
                trust_remote_code=True,
            )
            self.pipe = pipeline(
                task="automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                torch_dtype=TORCH_DTYPE,
                device=0 if DEVICE == "cuda" else -1,
            )

    def transcribe_pcm16le(self, pcm_bytes: bytes, sample_rate: int = CHUNK_SAMPLE_RATE) -> str:
        if not pcm_bytes:
            return ""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return ""
        result = self.pipe({"array": audio, "sampling_rate": sample_rate})
        return result.get("text", "").strip()


class VLLMASREngine(BaseASREngine):
    def __init__(self, model_id: str = MODEL_ID):
        if importlib.util.find_spec("qwen_asr") is None:
            raise RuntimeError("qwen_asr is not installed. Install with: pip install -U 'qwen-asr[vllm]'")

        qwen_asr = importlib.import_module("qwen_asr")
        model_cls = getattr(qwen_asr, "Qwen3ASRModel", None)
        if model_cls is None:
            raise RuntimeError("qwen_asr.Qwen3ASRModel was not found. Update qwen-asr package.")

        self.model = self._build_llm(model_cls, model_id)

    def _build_llm(self, model_cls: Any, model_id: str) -> Any:
        llm_fn = getattr(model_cls, "LLM", None)
        if llm_fn is None:
            raise RuntimeError("Qwen3ASRModel.LLM was not found in qwen_asr package.")

        attempts = [
            {"model": model_id, "token": HF_TOKEN, "revision": MODEL_REVISION, "device": DEVICE},
            {"model": model_id, "token": HF_TOKEN, "revision": MODEL_REVISION},
            {"model": model_id},
            {"model_name": model_id},
        ]
        for kwargs in attempts:
            filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
            try:
                return llm_fn(**filtered_kwargs)
            except TypeError:
                continue

        llm_sig = str(inspect.signature(llm_fn))
        raise RuntimeError(f"Could not initialize Qwen3ASRModel.LLM with supported arguments. Signature: {llm_sig}")

    def _extract_text(self, result: Any) -> str:
        if isinstance(result, str):
            return result.strip()
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return str(first.get("text", "")).strip()
            return str(first).strip()
        return ""

    def transcribe_pcm16le(self, pcm_bytes: bytes, sample_rate: int = CHUNK_SAMPLE_RATE) -> str:
        if not pcm_bytes:
            return ""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if audio.size == 0:
            return ""

        if hasattr(self.model, "transcribe"):
            try:
                result = self.model.transcribe(audio=audio, sampling_rate=sample_rate)
            except TypeError:
                result = self.model.transcribe(audio)
            return self._extract_text(result)

        if hasattr(self.model, "generate"):
            result = self.model.generate({"array": audio, "sampling_rate": sample_rate})
            return self._extract_text(result)

        raise RuntimeError("Loaded qwen-asr backend does not expose transcribe/generate API.")


def _try_build(backend: str, model_id: str) -> BaseASREngine:
    if backend == "vllm":
        return VLLMASREngine(model_id=model_id)
    if backend == "transformers":
        return TransformersASREngine(model_id=model_id)
    raise ValueError("Unsupported ASR_BACKEND. Use 'auto', 'vllm', or 'transformers'.")


def _vllm_usable() -> tuple[bool, str]:
    if DEVICE != "cuda":
        return False, "CUDA not available; skipping vllm backend."
    if not torch.cuda.is_available():
        return False, "torch.cuda.is_available() is false; skipping vllm backend."
    return True, ""


def create_engine(model_id: str = MODEL_ID, backend: str = ASR_BACKEND) -> tuple[BaseASREngine, str, Optional[str]]:
    if backend == "auto":
        usable, reason = _vllm_usable()
        if usable:
            try:
                return _try_build("vllm", model_id), "vllm", None
            except Exception as vllm_exc:
                engine = _try_build("transformers", model_id)
                return engine, "transformers", f"vllm initialization failed, fell back to transformers: {vllm_exc}"

        engine = _try_build("transformers", model_id)
        return engine, "transformers", reason

    if backend == "vllm":
        usable, reason = _vllm_usable()
        if not usable:
            raise RuntimeError(reason + " Set ASR_BACKEND=transformers or enable GPU runtime.")

    return _try_build(backend, model_id), backend, None


async def transcribe_with_timeout(audio_bytes: bytes, sample_rate: int) -> str:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(asr_engine.transcribe_pcm16le, audio_bytes, sample_rate),
            timeout=CHUNK_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Chunk transcription exceeded {CHUNK_TIMEOUT_SECONDS:.0f}s timeout."
        ) from exc


@dataclass
class StreamState:
    chunks: List[bytes] = field(default_factory=list)
    sample_rate: int = CHUNK_SAMPLE_RATE
    partial_every_chunks: int = 8

    def append(self, data: bytes) -> None:
        self.chunks.append(data)

    def all_audio(self) -> bytes:
        return b"".join(self.chunks)

    def should_emit_partial(self) -> bool:
        return len(self.chunks) % self.partial_every_chunks == 0


app = FastAPI(title="Qwen ASR Streaming API")
asr_engine: Optional[BaseASREngine] = None
startup_error: Optional[str] = None
startup_warning: Optional[str] = None
active_backend: Optional[str] = None


@app.on_event("startup")
def startup_event() -> None:
    global asr_engine, startup_error, startup_warning, active_backend
    try:
        asr_engine, active_backend, startup_warning = create_engine()
        startup_error = None
    except Exception as exc:
        asr_engine = None
        active_backend = None
        startup_warning = None
        startup_error = str(exc)


@app.get("/health")
def health() -> dict:
    if asr_engine is None:
        return {
            "status": "degraded",
            "model": MODEL_ID,
            "backend": ASR_BACKEND,
            "active_backend": active_backend,
            "device": DEVICE,
            "ready": False,
            "error": startup_error,
            "hint": "Set ASR_BACKEND=transformers or install qwen-asr[vllm], and verify MODEL_ID/HF_TOKEN.",
        }
    return {
        "status": "ok",
        "model": MODEL_ID,
        "backend": ASR_BACKEND,
        "active_backend": active_backend,
        "device": DEVICE,
        "ready": True,
        "warning": startup_warning,
    }


@app.post("/v1/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    if asr_engine is None:
        return JSONResponse(status_code=503, content={"error": "ASR model not initialized", "details": startup_error})

    data = await file.read()
    try:
        text = await asyncio.wait_for(asyncio.to_thread(asr_engine.transcribe_wav, data), timeout=CHUNK_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=504,
            content={"error": f"Transcription exceeded {CHUNK_TIMEOUT_SECONDS:.0f}s timeout."},
        )
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    return {"text": text}


@app.websocket("/v1/stream")
async def stream_transcribe(websocket: WebSocket):
    await websocket.accept()
    if asr_engine is None:
        await websocket.send_json({"type": "error", "message": "ASR model not initialized", "details": startup_error})
        await websocket.close(code=1011)
        return

    state = StreamState()
    await websocket.send_json(
        {
            "type": "ready",
            "audio_format": "pcm16le",
            "sample_rate": CHUNK_SAMPLE_RATE,
            "channels": CHUNK_CHANNELS,
            "backend": active_backend,
            "end_signal": "send text message: end",
        }
    )

    try:
        while True:
            message = await websocket.receive()
            if message.get("bytes") is not None:
                chunk = message["bytes"]
                if chunk:
                    state.append(chunk)
                    if state.should_emit_partial():
                        try:
                            partial = await transcribe_with_timeout(state.all_audio(), state.sample_rate)
                        except TimeoutError as exc:
                            await websocket.send_json({"type": "error", "message": str(exc)})
                            await websocket.close(code=1011)
                            return
                        await websocket.send_json({"type": "partial", "text": partial})
            elif message.get("text"):
                command = message["text"].strip().lower()
                if command == "end":
                    try:
                        final = await transcribe_with_timeout(state.all_audio(), state.sample_rate)
                    except TimeoutError as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
                        await websocket.close(code=1011)
                        return
                    await websocket.send_json({"type": "final", "text": final})
                    await websocket.close(code=1000)
                    return
                if command.startswith("sample_rate:"):
                    state.sample_rate = int(command.split(":", 1)[1])
                    await websocket.send_json({"type": "ack", "sample_rate": state.sample_rate})
                else:
                    await websocket.send_json({"type": "warning", "message": f"Unknown command: {command}"})
    except WebSocketDisconnect:
        return
