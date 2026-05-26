#!/usr/bin/env python3
"""Run full-sequence qualitative visualization for selected TNL2K sequences."""

import argparse
import csv
import importlib
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.config.salttrack.config import cfg, update_config_from_file
from lib.models.salttrack import build_salttrack
from lib.test.evaluation.environment import env_settings
from lib.test.evaluation.tnl2kdataset import TNL2kDataset
from lib.test.tracker.salttrack import get_resize_template_bbox
from lib.test.tracker.salttrack_utils import Preprocessor, sample_target, transform_image_to_crop
from lib.utils.ce_utils import generate_bbox_mask
from lib.utils.box_ops import clip_box


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize selected full TNL2K sequences.")
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--output", default="output/visualizations/selected_sequence_curves")
    parser.add_argument("--ours-yaml", default="salttrack_large_nobb_routed_both")
    parser.add_argument("--ours-checkpoint", default="output/checkpoints/train/salttrack/salttrack_large_nobb_routed_both/SALTTrack_ep0071.pth.tar")
    parser.add_argument("--base-yaml", default="salttrack_large_nobb_routed_both")
    parser.add_argument("--base-checkpoint", default="/data/MODEL_WEIGHTS_PUBLIC/VLT_weights/ATCTrack-master/models/ATCTrack_l.pth.tar")
    parser.add_argument("--base-result-root", default="", help="Optional baseline result txt root for boxes/IoU.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-seq", type=int, default=30)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means full sequence")
    parser.add_argument("--video-fps", type=int, default=12)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-every", type=int, default=20)
    return parser.parse_args()


def read_rgb(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_template_mask(template_bbox, resize_factor, template_size, device):
    resized_bbox = get_resize_template_bbox(template_bbox, resize_factor)
    resized_bbox = [torch.tensor(resized_bbox, device=device)]
    bbox_mask = torch.zeros((1, template_size, template_size), device=device)
    bbox_mask = generate_bbox_mask(bbox_mask, resized_bbox)
    bbox_mask = bbox_mask.unfold(1, 16, 16).unfold(2, 16, 16)
    return bbox_mask.mean(dim=(-1, -2)).view(bbox_mask.shape[0], -1).unsqueeze(-1)


def load_model(yaml_name, checkpoint, device, disable_lora=False):
    env = env_settings()
    update_config_from_file(str(Path(env.prj_dir) / "experiments" / "salttrack" / f"{yaml_name}.yaml"))
    if disable_lora:
        cfg.MODEL.LORA.ENABLED = False
    model = build_salttrack(cfg, training=False)
    state = torch.load(checkpoint, map_location="cpu")
    state_dict = state["net"] if isinstance(state, dict) and "net" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"{yaml_name} missing keys: {len(missing)}")
    if unexpected:
        print(f"{yaml_name} unexpected keys: {len(unexpected)}")
    model.to(device).eval()
    return model


def initialize(model, preprocessor, image, init_bbox, init_nlp, device):
    z_patch, resize_factor = sample_target(image, init_bbox, cfg.TEST.TEMPLATE_FACTOR, output_sz=cfg.TEST.TEMPLATE_SIZE)
    template = preprocessor.process(z_patch)
    bbox_mask = build_template_mask(init_bbox, resize_factor, cfg.TEST.TEMPLATE_SIZE, device)
    text_features, text_subject_features, _, _ = model.forward_text([init_nlp], 1, None, device=template.device)
    return {
        "template_list": [template] * cfg.TEST.NUM_TEMPLATES,
        "soft_token_template_mask": [bbox_mask, bbox_mask],
        "text_features": text_features,
        "text_subject_features": text_subject_features,
        "temporal_infor": [],
        "state": list(init_bbox),
        "first_frame_flag": True,
    }


def map_box_back(pred_box, prev_state, search_size, resize_factor):
    cx_prev = prev_state[0] + 0.5 * prev_state[2]
    cy_prev = prev_state[1] + 0.5 * prev_state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def xywh_iou(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    union = max(aw, 0.0) * max(ah, 0.0) + max(bw, 0.0) * max(bh, 0.0) - inter
    return float(inter / max(union, 1e-8))


def load_result_boxes(root, seq_name):
    if not root:
        return None
    path = Path(root) / f"{seq_name}.txt"
    if not path.is_file():
        return None
    try:
        boxes = np.loadtxt(path, delimiter="\t")
    except ValueError:
        boxes = np.loadtxt(path, delimiter=",")
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    return boxes[:, :4].astype(np.float64)


def draw_box(image, box, color, label):
    x, y, w, h = [int(round(v)) for v in box]
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(image, label, (x, max(y - 6, 16)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def track_one_frame(model, state, preprocessor, image, gt_box, device):
    height, width, _ = image.shape
    prev_state = list(state["state"])
    x_patch, resize_factor = sample_target(image, prev_state, cfg.TEST.SEARCH_FACTOR, output_sz=cfg.TEST.SEARCH_SIZE)
    search = preprocessor.process(x_patch)
    gt_xywh = torch.tensor(gt_box, dtype=torch.float32, device=device)
    crop_gt_xywh = transform_image_to_crop(
        gt_xywh,
        torch.tensor(prev_state, dtype=torch.float32, device=device),
        resize_factor,
        torch.tensor([cfg.TEST.SEARCH_SIZE, cfg.TEST.SEARCH_SIZE], dtype=torch.float32, device=device),
        normalize=True,
    ).clamp(min=0.0, max=1.0).view(1, 4)

    with torch.no_grad():
        out = model(
            state["template_list"],
            search,
            state["soft_token_template_mask"],
            exp_str=state["text_features"],
            exp_subject_mask=state["text_subject_features"],
            search_anno=crop_gt_xywh,
            temporal_infor=state["temporal_infor"],
            first_frame_flag=state["first_frame_flag"],
            training=False,
            return_visualization=True,
        )
    state["first_frame_flag"] = False
    state["temporal_infor"] = out["temporal_infor"]
    pred_boxes, score = model.box_head.cal_bbox(out["score_map"], out["size_map"], out["offset_map"], return_score=True)
    pred_norm = pred_boxes.view(-1, 4).mean(dim=0)
    pred_crop = (pred_norm * cfg.TEST.SEARCH_SIZE / resize_factor).detach().cpu().tolist()
    pred_global = clip_box(map_box_back(pred_crop, prev_state, cfg.TEST.SEARCH_SIZE, resize_factor), height, width, margin=10)
    state["state"] = pred_global

    pred_feat = out.get("pred_visual_features")
    gt_feat = out.get("gt_visual_features")
    text_feat = out.get("target_text_features")
    pred_text = float(F.cosine_similarity(pred_feat, text_feat).mean().item())
    gt_text = float(F.cosine_similarity(gt_feat, text_feat).mean().item())
    pred_gt = float(F.cosine_similarity(pred_feat, gt_feat).mean().item())
    pred_to_text = F.normalize(text_feat - pred_feat, dim=-1)
    gt_to_text = F.normalize(text_feat - gt_feat, dim=-1)
    direction_consistency = float(F.cosine_similarity(pred_to_text, gt_to_text, dim=-1).mean().item())
    semantic_margin = gt_text - pred_text
    semantic_gate = float(torch.sigmoid(torch.tensor(semantic_margin / 0.05)).item())
    return {
        "box": pred_global,
        "score": float(score.view(-1)[0].item()),
        "pred_text": pred_text,
        "gt_text": gt_text,
        "pred_gt": pred_gt,
        "direction_consistency": direction_consistency,
        "semantic_margin": semantic_margin,
        "semantic_gate": semantic_gate,
    }


def plot_sequence_curves(seq_dir, seq_name, rows):
    frames = [int(r["frame"]) for r in rows]
    ours_iou = [float(r["ours_iou"]) for r in rows]
    base_iou = [float(r["baseline_iou"]) for r in rows]
    ours_pred_text = [float(r["ours_pred_text_cos"]) for r in rows]
    base_pred_text = [float(r["baseline_pred_text_cos"]) for r in rows]
    ours_gt_text = [float(r["ours_gt_text_cos"]) for r in rows]
    base_gt_text = [float(r["baseline_gt_text_cos"]) for r in rows]
    ours_direction = [float(r["ours_direction_consistency"]) for r in rows]
    base_direction = [float(r["baseline_direction_consistency"]) for r in rows]
    ours_direction_gap = [1.0 - v for v in ours_direction]
    base_direction_gap = [1.0 - v for v in base_direction]
    ours_text_distance = [1.0 - v for v in ours_pred_text]
    base_text_distance = [1.0 - v for v in base_pred_text]
    ours_gate = [float(r["ours_semantic_gate"]) for r in rows]
    base_gate = [float(r["baseline_semantic_gate"]) for r in rows]

    plt.figure(figsize=(9, 4.2))
    plt.plot(frames, ours_iou, label="SALTTrack IoU", color="#d64f4f", linewidth=1.8)
    plt.plot(frames, base_iou, label="ATCTrack IoU", color="#3f66c2", linewidth=1.5)
    plt.ylim(0, 1)
    plt.xlabel("frame")
    plt.ylabel("IoU")
    plt.title(seq_name)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(seq_dir / "iou_curve.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 4.2))
    plt.plot(frames, ours_direction, label="SALTTrack direction", color="#d64f4f", linewidth=1.8)
    plt.plot(frames, base_direction, label="ATCTrack direction", color="#3f66c2", linewidth=1.5)
    plt.ylim(0, 1)
    plt.xlabel("frame")
    plt.ylabel("direction consistency")
    plt.title(seq_name)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(seq_dir / "direction_consistency_curve.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 4.2))
    plt.plot(frames, ours_direction_gap, label="SALTTrack direction distance", color="#d64f4f", linewidth=1.8)
    plt.plot(frames, base_direction_gap, label="ATCTrack direction distance", color="#3f66c2", linewidth=1.5)
    plt.ylim(0, 2)
    plt.xlabel("frame")
    plt.ylabel("1 - direction consistency")
    plt.title(seq_name)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(seq_dir / "direction_distance_curve.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 4.2))
    plt.plot(frames, ours_pred_text, label="SALTTrack pred-text", color="#d64f4f", linewidth=1.8)
    plt.plot(frames, ours_gt_text, label="SALTTrack gt-text", color="#d64f4f", linewidth=1.2, linestyle="--")
    plt.plot(frames, base_pred_text, label="ATCTrack pred-text", color="#3f66c2", linewidth=1.5)
    plt.plot(frames, base_gt_text, label="ATCTrack gt-text", color="#3f66c2", linewidth=1.1, linestyle="--")
    plt.ylim(-1, 1)
    plt.xlabel("frame")
    plt.ylabel("text-visual cosine similarity")
    plt.title(seq_name)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(seq_dir / "text_similarity_curve.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 4.2))
    plt.plot(frames, ours_text_distance, label="SALTTrack text distance", color="#d64f4f", linewidth=1.8)
    plt.plot(frames, base_text_distance, label="ATCTrack text distance", color="#3f66c2", linewidth=1.5)
    plt.ylim(0, 2)
    plt.xlabel("frame")
    plt.ylabel("1 - pred-text cosine")
    plt.title(seq_name)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(seq_dir / "text_distance_curve.png", dpi=220)
    plt.close()

    plt.figure(figsize=(9, 4.2))
    plt.plot(frames, ours_gate, label="SALTTrack semantic gate", color="#d64f4f", linewidth=1.8)
    plt.plot(frames, base_gate, label="ATCTrack semantic gate", color="#3f66c2", linewidth=1.5)
    plt.ylim(0, 1)
    plt.xlabel("frame")
    plt.ylabel("relative semantic gate")
    plt.title(seq_name)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(seq_dir / "semantic_gate_curve.png", dpi=220)
    plt.close()


def plot_direction_alignment_summary(seq_dir, seq_name, rows):
    ours_direction = np.asarray([float(r["ours_direction_consistency"]) for r in rows], dtype=np.float64)
    base_direction = np.asarray([float(r["baseline_direction_consistency"]) for r in rows], dtype=np.float64)
    ours_text_distance = np.asarray([1.0 - float(r["ours_pred_text_cos"]) for r in rows], dtype=np.float64)
    base_text_distance = np.asarray([1.0 - float(r["baseline_pred_text_cos"]) for r in rows], dtype=np.float64)
    ours_iou = np.asarray([float(r["ours_iou"]) for r in rows], dtype=np.float64)
    base_iou = np.asarray([float(r["baseline_iou"]) for r in rows], dtype=np.float64)

    labels = ["Direction", "Text distance", "IoU"]
    ours_values = [np.nanmean(ours_direction), np.nanmean(ours_text_distance), np.nanmean(ours_iou)]
    base_values = [np.nanmean(base_direction), np.nanmean(base_text_distance), np.nanmean(base_iou)]

    x = np.arange(len(labels))
    width = 0.34
    plt.figure(figsize=(7.2, 4.2))
    plt.bar(x - width / 2, ours_values, width, label="SALTTrack", color="#d64f4f")
    plt.bar(x + width / 2, base_values, width, label="ATCTrack", color="#3f66c2")
    plt.xticks(x, labels)
    plt.ylabel("mean value")
    plt.title(seq_name)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(seq_dir / "semantic_alignment_summary.png", dpi=220)
    plt.close()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_names = [line.strip() for line in Path(args.sequences).read_text().splitlines() if line.strip()][: args.max_seq]
    dataset = {seq.name: seq for seq in TNL2kDataset().get_sequence_list()}
    preprocessor = Preprocessor()

    print("Loading SALTTrack...")
    ours = load_model(args.ours_yaml, args.ours_checkpoint, device, disable_lora=False)
    print("Loading ATCTrack baseline...")
    baseline = load_model(args.base_yaml, args.base_checkpoint, device, disable_lora=True)

    summary_rows = []
    for seq_name in seq_names:
        seq = dataset[seq_name]
        seq_dir = out_dir / seq_name
        frames_dir = seq_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        init_image = read_rgb(seq.frames[0])
        init_info = seq.init_info()
        ours_state = initialize(ours, preprocessor, init_image, init_info["init_bbox"], init_info.get("init_nlp", ""), device)
        base_state = initialize(baseline, preprocessor, init_image, init_info["init_bbox"], init_info.get("init_nlp", ""), device)
        base_result_boxes = load_result_boxes(args.base_result_root, seq_name)
        max_frames = len(seq.frames) if args.max_frames <= 0 else min(len(seq.frames), args.max_frames)
        rows = []
        video_writer = None

        for frame_id in range(max_frames):
            image = read_rgb(seq.frames[frame_id])
            gt = seq.ground_truth_rect[frame_id].tolist()
            if frame_id == 0:
                ours_box = list(init_info["init_bbox"])
                base_box = list(init_info["init_bbox"])
                ours_stats = base_stats = {
                    "score": 1.0,
                    "pred_text": np.nan,
                    "gt_text": np.nan,
                    "pred_gt": np.nan,
                    "direction_consistency": np.nan,
                    "semantic_margin": np.nan,
                    "semantic_gate": np.nan,
                }
            else:
                ours_stats = track_one_frame(ours, ours_state, preprocessor, image, gt, device)
                base_stats = track_one_frame(baseline, base_state, preprocessor, image, gt, device)
                ours_box = ours_stats["box"]
                model_base_box = base_stats["box"]
                if base_result_boxes is not None and frame_id < len(base_result_boxes):
                    base_box = base_result_boxes[frame_id].tolist()
                else:
                    base_box = model_base_box

            canvas = image.copy()
            draw_box(canvas, gt, (70, 220, 70), "GT")
            draw_box(canvas, ours_box, (220, 70, 70), "SALT")
            draw_box(canvas, base_box, (70, 110, 230), "ATC")
            cv2.putText(
                canvas,
                f"f={frame_id} IoU salt={xywh_iou(ours_box, gt):.2f} atc={xywh_iou(base_box, gt):.2f}",
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if frame_id % max(1, args.save_every) == 0:
                cv2.imwrite(str(frames_dir / f"{frame_id:06d}.jpg"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            if args.save_video:
                if video_writer is None:
                    h, w = canvas.shape[:2]
                    video_writer = cv2.VideoWriter(
                        str(seq_dir / "comparison.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        args.video_fps,
                        (w, h),
                    )
                video_writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

            rows.append({
                "sequence": seq_name,
                "frame": frame_id,
                "ours_iou": xywh_iou(ours_box, gt),
                "baseline_iou": xywh_iou(base_box, gt),
                "ours_pred_text_cos": ours_stats["pred_text"],
                "baseline_pred_text_cos": base_stats["pred_text"],
                "ours_gt_text_cos": ours_stats["gt_text"],
                "baseline_gt_text_cos": base_stats["gt_text"],
                "ours_pred_gt_cos": ours_stats["pred_gt"],
                "baseline_pred_gt_cos": base_stats["pred_gt"],
                "ours_direction_consistency": ours_stats["direction_consistency"],
                "baseline_direction_consistency": base_stats["direction_consistency"],
                "ours_semantic_margin": ours_stats["semantic_margin"],
                "baseline_semantic_margin": base_stats["semantic_margin"],
                "ours_semantic_gate": ours_stats["semantic_gate"],
                "baseline_semantic_gate": base_stats["semantic_gate"],
                "ours_score": ours_stats["score"],
                "baseline_score": base_stats["score"],
            })

        if video_writer is not None:
            video_writer.release()
        with open(seq_dir / "curves.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        plot_sequence_curves(seq_dir, seq_name, rows[1:])
        plot_direction_alignment_summary(seq_dir, seq_name, rows[1:])
        summary_rows.append({
            "sequence": seq_name,
            "frames": len(rows),
            "ours_mean_iou": float(np.mean([r["ours_iou"] for r in rows[1:]])),
            "baseline_mean_iou": float(np.mean([r["baseline_iou"] for r in rows[1:]])),
            "gain_mean_iou": float(np.mean([r["ours_iou"] - r["baseline_iou"] for r in rows[1:]])),
            "ours_mean_direction": float(np.nanmean([r["ours_direction_consistency"] for r in rows[1:]])),
            "baseline_mean_direction": float(np.nanmean([r["baseline_direction_consistency"] for r in rows[1:]])),
            "ours_mean_semantic_gate": float(np.nanmean([r["ours_semantic_gate"] for r in rows[1:]])),
            "baseline_mean_semantic_gate": float(np.nanmean([r["baseline_semantic_gate"] for r in rows[1:]])),
        })
        print(seq_name, summary_rows[-1])

    with open(out_dir / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
