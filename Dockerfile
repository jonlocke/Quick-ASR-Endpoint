# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=0

ARG TORCH_SPEC="torch==2.4.1"
ARG TORCH_INDEX_URL=""
ARG TRANSFORMERS_SPEC="transformers==4.57.6"
ARG QWEN_ASR_SPEC="qwen-asr[vllm]"

LABEL org.opencontainers.image.title="quick-asr-endpoint"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip \
    && if [ -n "${TORCH_INDEX_URL}" ]; then \
         python -m pip install \
           "${TORCH_SPEC}" \
           "${QWEN_ASR_SPEC}" \
           "${TRANSFORMERS_SPEC}" \
           -r requirements.txt \
           --index-url "${TORCH_INDEX_URL}"; \
       else \
         python -m pip install \
           "${TORCH_SPEC}" \
           "${QWEN_ASR_SPEC}" \
           "${TRANSFORMERS_SPEC}" \
           -r requirements.txt; \
       fi

COPY app.py ./

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
