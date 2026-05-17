#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/wyp/SALTTrack}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/output}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OPENCV_OPENCL_RUNTIME=disabled

cd "${ROOT_DIR}"

"${PYTHON_BIN}" -m torch.distributed.launch --nproc_per_node "${NUM_GPUS}" \
  lib/train/run_training.py \
  --script salttrack \
  --config salttrack_base \
  --save_dir "${SAVE_DIR}" \
  --use_lmdb 0 \
  --use_wandb 0
