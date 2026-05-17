# SALTTrack Commands

## Train

```bash
bash scripts/train_salttrack.sh
```

## Evaluate

```bash
DATASET_NAME=tnl2k \
CKPT_PATH=/data/wyp/SALTTrack/output/checkpoints/train/salttrack/salttrack_base/SALTTrack_ep0080.pth.tar \
bash scripts/eval_salttrack.sh
```

## Analyze

```bash
python tracking/analysis_results.py \
  --tracker_name salttrack \
  --tracker_param salttrack_base \
  --dataset_name tnl2k
```
