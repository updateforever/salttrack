#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/data/wyp/SALTTrack}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_NAME="${DATASET_NAME:-tnl2k}"
CKPT_PATH="${CKPT_PATH:-${ROOT_DIR}/output/checkpoints/train/salttrack/salttrack_base/SALTTrack_ep0080.pth.tar}"
CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_GPUS="${NUM_GPUS:-1}"
THREADS="${THREADS:-4}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export OPENCV_OPENCL_RUNTIME=disabled

cd "${ROOT_DIR}"

"${PYTHON_BIN}" tracking/test.py \
  --tracker_name salttrack \
  --tracker_param salttrack_base \
  --dataset_name "${DATASET_NAME}" \
  --threads "${THREADS}" \
  --num_gpus "${NUM_GPUS}" \
  --ckpt_path "${CKPT_PATH}"
