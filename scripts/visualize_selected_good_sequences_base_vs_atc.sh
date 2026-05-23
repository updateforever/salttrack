#!/usr/bin/env bash
set -euo pipefail

cd /data/wyp/SALTTrack

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

/data/envs/wyp_vlt/bin/python tracking/visualization/visualize_selected_sequence_curves.py \
  --sequences output/visualizations/selected_good_sequences_base_vs_atcbase_official/selected_sequences.txt \
  --output output/visualizations/selected_good_sequences_base_vs_atcbase_official/full_sequence_curves \
  --ours-yaml salttrack_base_nobb_routed_both \
  --ours-checkpoint output/checkpoints/train/salttrack/salttrack_base_nobb_routed_both/SALTTrack_ep0078.pth.tar \
  --base-yaml salttrack_base_nobb_routed_both \
  --base-checkpoint /data/MODEL_WEIGHTS_PUBLIC/VLT_weights/ATCTrack-master/models/ATCTrack_b.pth.tar \
  --base-result-root /data/wyp/SALT-Track/output/atctrack_base/tnl2k/tnl2k \
  --device cuda:0 \
  --max-seq 30 \
  --max-frames 0 \
  --save-every 10 \
  --save-video
