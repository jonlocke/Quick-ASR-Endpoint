#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-qwen-asr-api:latest}"
CONTAINER_NAME="${2:-qwen-asr-api}"
PORT="${PORT:-8000}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-ASR-0.6B}"
ASR_BACKEND="${ASR_BACKEND:-vllm}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not on PATH."
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required by scripts/run.sh for health checks."
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
  -e ASR_BACKEND="${ASR_BACKEND}" \
  -p "${PORT}:8000" \
  "${IMAGE_NAME}")"

echo "Container created: ${CONTAINER_NAME} (${CONTAINER_ID})"
echo "Waiting for health endpoint: http://localhost:${PORT}/health (timeout: ${STARTUP_TIMEOUT}s)"

elapsed=0
while [ "${elapsed}" -lt "${STARTUP_TIMEOUT}" ]; do
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' exited before becoming healthy. Showing recent logs:"
    docker logs --tail 100 "${CONTAINER_NAME}" || true
    exit 1
  fi

  if curl -fsS "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "Container is healthy: ${CONTAINER_NAME} (${CONTAINER_ID})"
    echo "Health: http://localhost:${PORT}/health"
    exit 0
  fi

  sleep "${POLL_INTERVAL}"
  elapsed=$((elapsed + POLL_INTERVAL))
done

echo "Timed out waiting for health endpoint after ${STARTUP_TIMEOUT}s."
echo "Container is still running, showing last 100 log lines for diagnostics:"
docker logs --tail 100 "${CONTAINER_NAME}" || true
exit 1
