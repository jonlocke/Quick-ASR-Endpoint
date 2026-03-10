#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-quick-asr-endpoint}"

docker rm -f "${CONTAINER_NAME}"
