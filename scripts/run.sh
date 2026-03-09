#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-qwen-asr-api:latest}"
CONTAINER_NAME="${2:-qwen-asr-api}"
PORT="${PORT:-8000}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-ASR-0.6B}"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Removing existing container ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting ${CONTAINER_NAME} on port ${PORT}"
docker run -d \
  --name "${CONTAINER_NAME}" \
  -e MODEL_ID="${MODEL_ID}" \
  -p "${PORT}:8000" \
  "${IMAGE_NAME}"

echo "Container started. Health: http://localhost:${PORT}/health"
