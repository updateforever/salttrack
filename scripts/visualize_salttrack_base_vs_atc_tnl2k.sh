#!/usr/bin/env bash
set -euo pipefail

cd /data/wyp/SALTTrack

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

OUT_DIR=output/visualizations/salttrack_base_vs_atc_tnl2k
LOG_DIR=output/visualizations/logs
mkdir -p "$OUT_DIR" "$LOG_DIR"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
if [[ "${#GPUS[@]}" -lt 2 ]]; then
  echo "Need at least 2 GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}" >&2
  exit 1
fi

run_job() {
  local gpu="$1"
  local log_file="$2"
  shift 2
  echo "$(date '+%F %T') start GPU ${gpu}: $log_file"
  CUDA_VISIBLE_DEVICES="${gpu}" /data/envs/wyp_vlt/bin/python "$@" > "${log_file}" 2>&1
}

run_job "${GPUS[0]}" "$LOG_DIR/salttrack_base_vs_atc_salttrack.log" \
  tracking/visualization/export_salttrack_semantic_visualizations.py \
  --yaml salttrack_base_nobb_routed_both \
  --best-metric-csv output/eval_logs/nobb_routed_tnl2k_metrics/tnl2k_salttrack_base_nobb_routed_both_metrics_from70.csv \
  --display-name SALTTrack-Base \
  --dataset tnl2k \
  --output "$OUT_DIR" \
  --max-seq 350 \
  --frames-per-seq 8 \
  --frame-stride 20 \
  --heatmaps 180 \
  --device cuda:0 &
PID_SALT=$!

run_job "${GPUS[1]}" "$LOG_DIR/salttrack_base_vs_atc_atctrack.log" \
  tracking/visualization/export_salttrack_semantic_visualizations.py \
  --yaml salttrack_base_nobb_routed_both \
  --checkpoint /data/MODEL_WEIGHTS_PUBLIC/VLT_weights/ATCTrack-master/models/ATCTrack_b.pth.tar \
  --display-name ATCTrack-Base \
  --disable-lora \
  --dataset tnl2k \
  --output "$OUT_DIR" \
  --max-seq 350 \
  --frames-per-seq 8 \
  --frame-stride 20 \
  --heatmaps 180 \
  --device cuda:0 &
PID_ATC=$!

wait "$PID_SALT"
wait "$PID_ATC"

/data/envs/wyp_vlt/bin/python tracking/visualization/plot_salttrack_visualization_comparison.py \
  --output "$OUT_DIR/comparison" \
  "$OUT_DIR"/SALTTrack-Base_ep*/semantic_samples.csv \
  "$OUT_DIR"/ATCTrack-Base_external/semantic_samples.csv

echo "$(date '+%F %T') done: $OUT_DIR"
