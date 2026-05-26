#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data/wyp/SALTTrack"
PYTHON_BIN="/data/envs/wyp_vlt/bin/python"
SAVE_DIR="${ROOT_DIR}/output"
LOG_DIR="${SAVE_DIR}/train_logs/nobb_routed_dircon_hard_limited_8gpu"
CUDA_DEVICES="0,1,2,3,4,5,6,7"
NUM_GPUS=8

CONFIGS=(
  "salttrack_base_nobb_routed_dircon_hard_limited"
  "salttrack_large_nobb_routed_dircon_hard_limited"
)

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export OPENCV_OPENCL_RUNTIME=disabled
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

mkdir -p "${LOG_DIR}"
cd "${ROOT_DIR}"

for CONFIG in "${CONFIGS[@]}"; do
  CONFIG_FILE="${ROOT_DIR}/experiments/salttrack/${CONFIG}.yaml"
  LOG_FILE="${LOG_DIR}/${CONFIG}.train.log"

  if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "[ERROR] Missing config: ${CONFIG_FILE}" | tee -a "${LOG_DIR}/train_salttrack_nobb_routed_dircon_hard_limited_8gpu.log"
    exit 1
  fi

  echo "================================================================" | tee -a "${LOG_DIR}/train_salttrack_nobb_routed_dircon_hard_limited_8gpu.log"
  echo "[START] ${CONFIG} $(date '+%F %T')" | tee -a "${LOG_DIR}/train_salttrack_nobb_routed_dircon_hard_limited_8gpu.log"
  echo "[LOG] ${LOG_FILE}" | tee -a "${LOG_DIR}/train_salttrack_nobb_routed_dircon_hard_limited_8gpu.log"

  "${PYTHON_BIN}" -m torch.distributed.launch --nproc_per_node "${NUM_GPUS}" \
    lib/train/run_training.py \
    --script salttrack \
    --config "${CONFIG}" \
    --save_dir "${SAVE_DIR}" \
    --use_lmdb 0 \
    --use_wandb 0 2>&1 | tee "${LOG_FILE}"

  echo "[DONE] ${CONFIG} $(date '+%F %T')" | tee -a "${LOG_DIR}/train_salttrack_nobb_routed_dircon_hard_limited_8gpu.log"
done

echo "[ALL DONE] $(date '+%F %T')" | tee -a "${LOG_DIR}/train_salttrack_nobb_routed_dircon_hard_limited_8gpu.log"
