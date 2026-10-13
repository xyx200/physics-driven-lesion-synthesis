#!/usr/bin/env bash
set -euo pipefail

INPUT=${1:?usage: run_task180_prediction.sh INPUT OUTPUT TASK [GPU]}
OUTPUT=${2:?output directory}
TASK=${3:?nnUNet task name or ID}
GPU=${4:-0}

CUDA_VISIBLE_DEVICES="$GPU" mednextv1_predict \
  -i "$INPUT" \
  -o "$OUTPUT" \
  -t "$TASK" \
  -tr nnUNetTrainerV2_MedNeXt_M_kernel3 \
  -m 3d_fullres \
  -p nnUNetPlansv2.1_trgSp_1x1x1 \
  -f 0 1 2 3 4
