FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG TORCH_SPEC="torch==2.4.1"
ARG TORCH_INDEX_URL=""
ARG TRANSFORMERS_SPEC="git+https://github.com/huggingface/transformers.git"
ARG QWEN_ASR_SPEC="qwen-asr[vllm]"

LABEL org.opencontainers.image.title="quick-asr-endpoint"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && if [ -n "${TORCH_INDEX_URL}" ]; then \
         pip install "${TORCH_SPEC}" --index-url "${TORCH_INDEX_URL}"; \
       else \
         pip install "${TORCH_SPEC}"; \
       fi \
    && pip install "${TRANSFORMERS_SPEC}" \
    && pip install "${QWEN_ASR_SPEC}" \
    && pip install -r requirements.txt

COPY app.py ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
