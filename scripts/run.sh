#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-quick-asr-endpoint:latest}"
CONTAINER_NAME="${2:-quick-asr-endpoint}"
PORT="${PORT:-8000}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-ASR-0.6B}"
ASR_BACKEND="${ASR_BACKEND:-vllm}"
ENABLE_GPU="${ENABLE_GPU:-auto}"
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"
HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface/quick-asr-endpoint}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
POLL_INTERVAL="${POLL_INTERVAL:-2}"
EXPECTED_IMAGE_LABEL="quick-asr-endpoint"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not on PATH."
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "Error: curl is required by scripts/run.sh for health checks."
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Error: python is required by scripts/run.sh to parse /health readiness."
  exit 1
fi

if ! docker image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Error: Docker image '${IMAGE_NAME}' not found. Build it first with ./scripts/build.sh ${IMAGE_NAME}."
  exit 1
fi

IMAGE_LABEL="$(docker image inspect "${IMAGE_NAME}" --format '{{ index .Config.Labels "org.opencontainers.image.title" }}' 2>/dev/null || true)"
if [ "${IMAGE_LABEL}" != "${EXPECTED_IMAGE_LABEL}" ]; then
  echo "Error: image '${IMAGE_NAME}' does not look like Quick-ASR-Endpoint (label org.opencontainers.image.title=${IMAGE_LABEL:-<missing>})."
  echo "Hint: rebuild with ./scripts/build.sh ${IMAGE_NAME} or use the correct image tag."
  exit 1
fi

GPU_ARGS=()
if [ "${ENABLE_GPU}" = "1" ] || [ "${ENABLE_GPU}" = "true" ]; then
  GPU_ARGS=(--gpus all)
elif [ "${ENABLE_GPU}" = "auto" ] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
fi

if [ "${ASR_BACKEND}" = "vllm" ] && [ ${#GPU_ARGS[@]} -eq 0 ]; then
  echo "Warning: ASR_BACKEND=vllm but GPU runtime is not enabled; startup may fail."
  echo "Hint: set ENABLE_GPU=1 and ensure Docker NVIDIA runtime is configured."
fi

mkdir -p "${HF_CACHE_DIR}"

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  echo "Removing existing container ${CONTAINER_NAME}"
  docker rm -f "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting ${CONTAINER_NAME} on port ${PORT} (backend=${ASR_BACKEND}, gpu=${ENABLE_GPU}, restart=${RESTART_POLICY})"
echo "Using HF cache mount: ${HF_CACHE_DIR} -> /root/.cache/huggingface"
CONTAINER_ID="$(docker run -d \
  --name "${CONTAINER_NAME}" \
  --restart "${RESTART_POLICY}" \
  -e MODEL_ID="${MODEL_ID}" \
  -e ASR_BACKEND="${ASR_BACKEND}" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  "${GPU_ARGS[@]}" \
  -p "${PORT}:8000" \
  "${IMAGE_NAME}")"

echo "Container created: ${CONTAINER_NAME} (${CONTAINER_ID})"
echo "Waiting for health endpoint: http://localhost:${PORT}/health (timeout: ${STARTUP_TIMEOUT}s)"

elapsed=0
while [ "${elapsed}" -lt "${STARTUP_TIMEOUT}" ]; do
  if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Container '${CONTAINER_NAME}' exited before becoming ready. Showing recent logs:"
    docker logs --tail 100 "${CONTAINER_NAME}" || true
    exit 1
  fi

  if HEALTH_JSON="$(curl -fsS "http://localhost:${PORT}/health" 2>/dev/null)"; then
    if python -c 'import json,sys; d=json.loads(sys.argv[1]); sys.exit(0 if d.get("ready") is True else 1)' "${HEALTH_JSON}"; then
      echo "Container is healthy: ${CONTAINER_NAME} (${CONTAINER_ID})"
      echo "Health: http://localhost:${PORT}/health"
      exit 0
    fi
  fi

  sleep "${POLL_INTERVAL}"
  elapsed=$((elapsed + POLL_INTERVAL))
done

echo "Timed out waiting for ready /health after ${STARTUP_TIMEOUT}s."
echo "Container is still running, showing last 100 log lines for diagnostics:"
docker logs --tail 100 "${CONTAINER_NAME}" || true
exit 1
