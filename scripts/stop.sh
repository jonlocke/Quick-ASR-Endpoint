#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-qwen-asr-api}"

docker rm -f "${CONTAINER_NAME}"
