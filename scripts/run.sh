#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-qwen-asr-api:latest}"
CONTAINER_NAME="${2:-qwen-asr-api}"
PORT="${PORT:-8000}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-ASR-0.6B}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not on PATH."
  exit 1
fi

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Error: Docker image '${IMAGE_NAME}' not found. Build it first with ./scripts/build.sh ${IMAGE_NAME}."
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Removing existing container ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting ${CONTAINER_NAME} on port ${PORT}"
CONTAINER_ID="$(docker run -d \
  --name "${CONTAINER_NAME}" \
  -e MODEL_ID="${MODEL_ID}" \
  -p "${PORT}:8000" \
  "${IMAGE_NAME}")"

sleep 2
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Container started successfully: ${CONTAINER_NAME} (${CONTAINER_ID})"
  echo "Health: http://localhost:${PORT}/health"
  exit 0
fi

echo "Container '${CONTAINER_NAME}' exited immediately. Showing recent logs:"
docker logs --tail 100 "${CONTAINER_NAME}" || true
exit 1
