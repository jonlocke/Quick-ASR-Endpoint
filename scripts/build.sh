#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${1:-qwen-asr-api:latest}"
TORCH_SPEC="${TORCH_SPEC:-torch==2.4.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

echo "Building Docker image: ${IMAGE_NAME}"
echo "Using TORCH_SPEC=${TORCH_SPEC}"
if [ -n "${TORCH_INDEX_URL}" ]; then
  echo "Using TORCH_INDEX_URL=${TORCH_INDEX_URL}"
fi

docker build \
  --build-arg TORCH_SPEC="${TORCH_SPEC}" \
  --build-arg TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
  -t "${IMAGE_NAME}" .

echo "Done."
