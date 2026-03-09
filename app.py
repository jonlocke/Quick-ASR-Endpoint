import io
import os
import wave
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-ASR-0.6B")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_REVISION = os.getenv("MODEL_REVISION")
CHUNK_SAMPLE_RATE = 16000
CHUNK_WIDTH = 2  # int16
CHUNK_CHANNELS = 1


class ASREngine:
    def __init__(self, model_id: str = MODEL_ID):
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
asr_engine: Optional[ASREngine] = None
startup_error: Optional[str] = None


@app.on_event("startup")
def startup_event() -> None:
    global asr_engine, startup_error
    try:
        asr_engine = ASREngine()
        startup_error = None
    except Exception as exc:
        asr_engine = None
        startup_error = str(exc)


@app.get("/health")
def health() -> dict:
    if asr_engine is None:
        return {
            "status": "degraded",
            "model": MODEL_ID,
            "device": DEVICE,
            "ready": False,
            "error": startup_error,
            "hint": "Set MODEL_ID to a valid public model or provide HF_TOKEN for private/gated repos.",
        }
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE, "ready": True}


@app.post("/v1/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    if asr_engine is None:
        return JSONResponse(status_code=503, content={"error": "ASR model not initialized", "details": startup_error})

    data = await file.read()
    try:
        text = asr_engine.transcribe_wav(data)
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
                        partial = asr_engine.transcribe_pcm16le(state.all_audio(), state.sample_rate)
                        await websocket.send_json({"type": "partial", "text": partial})
            elif message.get("text"):
                command = message["text"].strip().lower()
                if command == "end":
                    final = asr_engine.transcribe_pcm16le(state.all_audio(), state.sample_rate)
                    await websocket.send_json({"type": "final", "text": final})
                    await websocket.close(code=1000)
                    return
                elif command.startswith("sample_rate:"):
                    state.sample_rate = int(command.split(":", 1)[1])
                    await websocket.send_json({"type": "ack", "sample_rate": state.sample_rate})
                else:
                    await websocket.send_json({"type": "warning", "message": f"Unknown command: {command}"})
    except WebSocketDisconnect:
        return
