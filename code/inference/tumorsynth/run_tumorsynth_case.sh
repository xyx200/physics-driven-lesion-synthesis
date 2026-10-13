#!/usr/bin/env bash
set -euo pipefail

RUNNER=${1:?usage: run_tumorsynth_case.sh RUNNER T1 FLAIR OUTPUT [GPU]}
T1=${2:?T1 image}
FLAIR=${3:?FLAIR image}
OUTPUT=${4:?output NIfTI file}
GPU=${5:-0}

CUDA_VISIBLE_DEVICES="$GPU" "$RUNNER" \
  --i "$T1,$FLAIR" --o "$OUTPUT" --wholetumor --threads 4
