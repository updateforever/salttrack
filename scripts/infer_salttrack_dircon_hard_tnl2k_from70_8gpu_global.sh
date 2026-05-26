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
  "salttrack_base_nobb_routed_dircon_hard"
  "salttrack_large_nobb_routed_dircon_hard"
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

run_tag_for_checkpoint() {
  local tracker_param="$1"
  local ckpt_path="$2"
  local ckpt_name
  local epoch

  ckpt_name="$(basename "${ckpt_path}")"
  epoch="$(echo "${ckpt_name}" | grep -oP 'ep\K[0-9]+')"
  echo "${tracker_param}_ep${epoch}"
}

is_done() {
  local tracker_param="$1"
  local ckpt_path="$2"
  local run_tag
  local infer_log
  local result_dir

  run_tag="$(run_tag_for_checkpoint "${tracker_param}" "${ckpt_path}")"
  infer_log="${ROOT_DIR}/output/eval_logs/${tracker_param}/${DATASET_NAME}/${run_tag}.infer.log"
  result_dir="${ROOT_DIR}/output/${tracker_param}/${run_tag}"

  [[ "${SKIP_DONE}" == "1" ]] && [[ -d "${result_dir}" ]] && [[ -f "${infer_log}" ]] && tail -5 "${infer_log}" | grep -q '^Done$'
}

run_one_checkpoint() {
  local tracker_param="$1"
  local gpu_id="$2"
  local ckpt_path="$3"
  local run_tag
  local log_dir
  local infer_log
  local failed_csv

  run_tag="$(run_tag_for_checkpoint "${tracker_param}" "${ckpt_path}")"
  log_dir="${ROOT_DIR}/output/eval_logs/${tracker_param}/${DATASET_NAME}"
  infer_log="${log_dir}/${run_tag}.infer.log"
  failed_csv="${ROOT_DIR}/output/eval_logs/${tracker_param}/failed_jobs.csv"
  mkdir -p "${log_dir}"

  if is_done "${tracker_param}" "${ckpt_path}"; then
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

declare -a tasks=()

for tracker_param in "${CONFIGS[@]}"; do
  checkpoint_dir="${ROOT_DIR}/output/checkpoints/train/${TRACKER_NAME}/${tracker_param}"
  mkdir -p "${ROOT_DIR}/output/eval_logs/${tracker_param}/${DATASET_NAME}"
  : > "${ROOT_DIR}/output/eval_logs/${tracker_param}/failed_jobs.csv"

  while IFS= read -r ckpt_path; do
    if is_done "${tracker_param}" "${ckpt_path}"; then
      run_tag="$(run_tag_for_checkpoint "${tracker_param}" "${ckpt_path}")"
      echo "[$(date '+%F %T')] config=${tracker_param} dataset=${DATASET_NAME} skip done ${run_tag}"
      continue
    fi
    tasks+=("${tracker_param}|${ckpt_path}")
  done < <(
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
done

echo "============================================================"
echo "Starting SALTTrack DirCon-Hard global 8-GPU checkpoint inference"
echo "CONFIGS=${CONFIGS[*]}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "EPOCH_RANGE=${START_EPOCH}-${END_EPOCH}"
echo "GPUS=${GPUS_CSV}"
echo "THREADS_PER_JOB=${THREADS_PER_JOB}"
echo "PENDING_TASKS=${#tasks[@]}"
echo "PYTHON_BIN=${PYTHON_BIN}"
echo "============================================================"

if [[ "${#tasks[@]}" -eq 0 ]]; then
  echo "No pending tasks."
  exit 0
fi

task_idx=0
failed_count=0
declare -a active_pids=()

for task in "${tasks[@]}"; do
  tracker_param="${task%%|*}"
  ckpt_path="${task#*|}"
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

echo "[$(date '+%F %T')] All pending tasks finished."
if [[ "${failed_count}" -gt 0 ]]; then
  echo "[$(date '+%F %T')] Finished with ${failed_count} failed jobs."
  exit 1
fi
