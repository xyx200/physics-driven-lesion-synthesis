#!/usr/bin/env bash
set -euo pipefail

MANIFEST=${1:?usage: run_brainseg.sh MANIFEST [GPU]}
GPU=${2:-0}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CUDA_VISIBLE_DEVICES="$GPU" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python "$SCRIPT_DIR/predict_brainseg_run.py" \
  --manifest "$MANIFEST" --mode lesion --device cuda:0
