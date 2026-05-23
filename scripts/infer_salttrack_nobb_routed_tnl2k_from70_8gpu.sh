#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/data/wyp/SALTTrack"
PYTHON_BIN="/data/envs/wyp_vlt/bin/python"
TRACKER_NAME="salttrack"
DATASET_NAME="tnl2k"
START_EPOCH=70
END_EPOCH=80
GPUS_CSV="0,1,2,3,4,5,6,7"
THREADS_PER_JOB=6
SKIP_DONE=1

CONFIGS=(
  "salttrack_base_nobb_routed_direction"
  "salttrack_base_nobb_routed_both"
  "salttrack_large_nobb_routed_direction"
  "salttrack_large_nobb_routed_both"
)

OMP_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
MKL_NUM_THREADS=4
VECLIB_MAXIMUM_THREADS=4
NUMEXPR_NUM_THREADS=4
OPENCV_OPENCL_RUNTIME=disabled
TOKENIZERS_PARALLELISM=false

cd "${ROOT_DIR}"

IFS=',' read -r -a GPU_LIST <<< "${GPUS_CSV}"
NUM_GPUS="${#GPU_LIST[@]}"

if [[ "${NUM_GPUS}" -eq 0 ]]; then
  echo "No GPUs provided."
  exit 1
fi

run_one_checkpoint() {
  local tracker_param="$1"
  local gpu_id="$2"
  local ckpt_path="$3"
  local ckpt_name
  local epoch
  local run_tag
  local log_dir
  local infer_log
  local result_dir
  local failed_csv

  ckpt_name="$(basename "${ckpt_path}")"
  epoch="$(echo "${ckpt_name}" | grep -oP 'ep\K[0-9]+')"
  run_tag="${tracker_param}_ep${epoch}"
  log_dir="${ROOT_DIR}/output/eval_logs/${tracker_param}/${DATASET_NAME}"
  infer_log="${log_dir}/${run_tag}.infer.log"
  result_dir="${ROOT_DIR}/output/${tracker_param}/${run_tag}"
  failed_csv="${ROOT_DIR}/output/eval_logs/${tracker_param}/failed_jobs.csv"
  mkdir -p "${log_dir}"

  if [[ "${SKIP_DONE}" == "1" ]] && [[ -d "${result_dir}" ]] && [[ -f "${infer_log}" ]] && tail -5 "${infer_log}" | grep -q '^Done$'; then
    echo "[$(date '+%F %T')] config=${tracker_param} dataset=${DATASET_NAME} skip done ${run_tag}"
    return 0
  fi

  echo "[$(date '+%F %T')] config=${tracker_param} dataset=${DATASET_NAME} GPU=${gpu_id} run=${run_tag}"

  if ! CUDA_VISIBLE_DEVICES="${gpu_id}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS}" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS}" \
    MKL_NUM_THREADS="${MKL_NUM_THREADS}" \
    VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS}" \
    NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS}" \
    OPENCV_OPENCL_RUNTIME="${OPENCV_OPENCL_RUNTIME}" \
    TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM}" \
    "${PYTHON_BIN}" tracking/test.py \
      --tracker_name "${TRACKER_NAME}" \
      --tracker_param "${tracker_param}" \
      --dataset_name "${DATASET_NAME}" \
      --ckpt_path "${ckpt_path}" \
      --threads "${THREADS_PER_JOB}" \
      --num_gpus 1 \
      --run_tag "${run_tag}" \
      > "${infer_log}" 2>&1; then
    echo "[$(date '+%F %T')] config=${tracker_param} dataset=${DATASET_NAME} GPU=${gpu_id} failed ${run_tag}; see ${infer_log}"
    echo "${DATASET_NAME},${run_tag},${ckpt_path},${infer_log}" >> "${failed_csv}"
    return 1
  fi

  echo "[$(date '+%F %T')] config=${tracker_param} dataset=${DATASET_NAME} GPU=${gpu_id} finished ${run_tag}"
}

run_one_config() {
  local tracker_param="$1"
  local checkpoint_dir="${ROOT_DIR}/output/checkpoints/train/${TRACKER_NAME}/${tracker_param}"
  local task_idx=0
  local failed_count=0
  local gpu_id
  local ckpt_path
  declare -a checkpoints=()
  declare -a active_pids=()

  mapfile -t checkpoints < <(
    find "${checkpoint_dir}" -maxdepth 1 -type f -name 'SALTTrack_ep*.pth.tar' \
      | sort -V \
      | while read -r path; do
          ckpt_name="$(basename "${path}")"
          epoch="$(echo "${ckpt_name}" | grep -oP 'ep\K[0-9]+')"
          if [[ -n "${epoch}" ]] && [[ $((10#${epoch})) -ge "${START_EPOCH}" ]] && [[ $((10#${epoch})) -le "${END_EPOCH}" ]]; then
            echo "${path}"
          fi
        done
  )

  if [[ "${#checkpoints[@]}" -eq 0 ]]; then
    echo "No checkpoints found in ${checkpoint_dir} from epoch ${START_EPOCH} to ${END_EPOCH}."
    return 1
  fi

  mkdir -p "${ROOT_DIR}/output/eval_logs/${tracker_param}/${DATASET_NAME}"
  : > "${ROOT_DIR}/output/eval_logs/${tracker_param}/failed_jobs.csv"

  echo "============================================================"
  echo "Starting SALTTrack 8-GPU checkpoint inference"
  echo "TRACKER_PARAM=${tracker_param}"
  echo "DATASET_NAME=${DATASET_NAME}"
  echo "CHECKPOINT_DIR=${checkpoint_dir}"
  echo "EPOCH_RANGE=${START_EPOCH}-${END_EPOCH}"
  echo "GPUS=${GPUS_CSV}"
  echo "THREADS_PER_JOB=${THREADS_PER_JOB}"
  echo "PYTHON_BIN=${PYTHON_BIN}"
  echo "============================================================"

  for ckpt_path in "${checkpoints[@]}"; do
    gpu_id="${GPU_LIST[$((task_idx % NUM_GPUS))]}"
    task_idx=$((task_idx + 1))

    while [[ "${#active_pids[@]}" -ge "${NUM_GPUS}" ]]; do
      if ! wait "${active_pids[0]}"; then
        failed_count=$((failed_count + 1))
      fi
      active_pids=("${active_pids[@]:1}")
    done

    run_one_checkpoint "${tracker_param}" "${gpu_id}" "${ckpt_path}" &
    active_pids+=("$!")
  done

  for pid in "${active_pids[@]}"; do
    if ! wait "${pid}"; then
      failed_count=$((failed_count + 1))
    fi
  done

  echo "[$(date '+%F %T')] Finished ${tracker_param}."
  if [[ "${failed_count}" -gt 0 ]]; then
    echo "[$(date '+%F %T')] ${tracker_param} finished with ${failed_count} failed jobs; see output/eval_logs/${tracker_param}/failed_jobs.csv"
    return 1
  fi
}

failed_configs=0
for tracker_param in "${CONFIGS[@]}"; do
  if ! run_one_config "${tracker_param}"; then
    failed_configs=$((failed_configs + 1))
  fi
done

if [[ "${failed_configs}" -gt 0 ]]; then
  echo "[$(date '+%F %T')] Finished with ${failed_configs} failed config groups."
  exit 1
fi

echo "[$(date '+%F %T')] All SALTTrack routed no-backbone checkpoint inference finished."
