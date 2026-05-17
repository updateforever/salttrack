# SALTTrack

SALTTrack is a visual-language tracking project built around text-driven semantic supervision and parameter-efficient adaptation. The core idea is to use text not only as an input condition, but also as a training-time supervision signal that encourages the tracker to learn instance features aligned with the target description. The default `salttrack_base` configuration uses semantic-guided LoRA with directional text supervision.

## Core Idea

SALTTrack augments standard tracking supervision with an instance-level semantic alignment loss:

```text
L_total = L_giou + L_l1 + L_focal + L_confidence + lambda(t) * L_semantic
```

The semantic loss combines visual consistency between predicted and ground-truth instance features with text-driven semantic guidance. The default configuration uses directional text supervision:

```text
L_text_direction = 1 - cos(normalize(text - pred), normalize(text - gt))
```

This direction-based formulation avoids forcing all instance features to collapse toward the text embedding while still guiding predictions toward the target semantics.

## Project Layout

- `lib/models/salttrack/`: SALTTrack model implementation.
- `lib/train/actors/salttrack.py`: training forward pass and losses.
- `lib/config/salttrack/`: default configuration schema.
- `experiments/salttrack/salttrack_base.yaml`: final semantic-guided LoRA experiment configuration.
- `tracking/test.py`: evaluation entry.
- `tracking/analysis_results.py`: result analysis entry.
- `scripts/`: clean training and evaluation commands.

## Train

```bash
bash scripts/train_salttrack.sh
```

Equivalent explicit command:

```bash
python -m torch.distributed.launch --nproc_per_node 4 \
  lib/train/run_training.py \
  --script salttrack \
  --config salttrack_base \
  --save_dir output \
  --use_lmdb 0 \
  --use_wandb 0
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

## Pretrained Resources

The project expects pretrained resources under:

```text
resource/pretrained_models/
```

In this local workspace, that path is a symlink to the existing pretrained resource directory to avoid duplicating large files.
