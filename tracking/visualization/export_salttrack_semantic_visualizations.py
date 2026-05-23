#!/usr/bin/env python3
"""Export semantic-alignment and routed-LoRA visualizations for SALTTrack.

The script intentionally samples only a small subset by default. It is meant to
produce paper/debug evidence quickly, not to evaluate the full benchmark.
"""

import argparse
import csv
import importlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from lib.config.salttrack.config import cfg, update_config_from_file
from lib.models.layers.lora import collect_lora_router_weights_named
from lib.models.salttrack import build_salttrack
from lib.test.evaluation.environment import env_settings
from lib.test.tracker.salttrack import get_resize_template_bbox
from lib.test.tracker.salttrack_utils import Preprocessor, sample_target, transform_image_to_crop
from lib.utils.ce_utils import generate_bbox_mask
from lib.utils.box_ops import box_cxcywh_to_xyxy, clip_box


@dataclass
class TrackerState:
    template_list: List[torch.Tensor]
    soft_token_template_mask: List[torch.Tensor]
    text_features: object
    text_subject_features: object
    temporal_infor: List[torch.Tensor]
    state: List[float]
    first_frame_flag: bool


def parse_args():
    parser = argparse.ArgumentParser(description="Export SALTTrack semantic visualization artifacts.")
    parser.add_argument("--yaml", required=True, help="Experiment yaml name under experiments/salttrack, without .yaml")
    parser.add_argument("--epoch", type=int, default=None, help="Checkpoint epoch")
    parser.add_argument("--best-metric-csv", default=None, help="Use the best AUC epoch from this metrics CSV")
    parser.add_argument("--checkpoint", default=None, help="Explicit checkpoint path. Overrides --epoch checkpoint lookup.")
    parser.add_argument("--display-name", default=None, help="Name used in output folder and plots.")
    parser.add_argument("--disable-lora", action="store_true", help="Build the model without LoRA wrappers before loading.")
    parser.add_argument("--dataset", default="tnl2k", choices=["tnl2k"])
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max-seq", type=int, default=20)
    parser.add_argument("--frames-per-seq", type=int, default=4)
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--heatmaps", type=int, default=24)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def resolve_epoch(args):
    if args.checkpoint is not None and args.epoch is None and args.best_metric_csv is None:
        return -1
    if args.epoch is not None:
        return args.epoch
    if args.best_metric_csv is None:
        raise ValueError("Either --epoch or --best-metric-csv must be provided.")
    with open(args.best_metric_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in metrics CSV: {args.best_metric_csv}")
    best = max(rows, key=lambda row: float(row["auc"]))
    match = re.search(r"ep(\d+)", best["run_tag"])
    if match is None:
        raise ValueError(f"Cannot parse epoch from run_tag: {best['run_tag']}")
    epoch = int(match.group(1))
    print(
        "Selected best epoch from {}: ep{:04d}, AUC {:.4f}, P {:.4f}, NP {:.4f}".format(
            args.best_metric_csv,
            epoch,
            float(best["auc"]),
            float(best["precision"]),
            float(best["norm_precision"]),
        )
    )
    return epoch


def load_sequence_list(dataset_name):
    if dataset_name != "tnl2k":
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    module = importlib.import_module("lib.test.evaluation.tnl2kdataset")
    return module.TNL2kDataset().get_sequence_list()


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


def load_model(yaml_name, epoch, device, checkpoint_override=None):
    env = env_settings()
    yaml_path = Path(env.prj_dir) / "experiments" / "salttrack" / f"{yaml_name}.yaml"
    update_config_from_file(str(yaml_path))
    if getattr(load_model, "disable_lora", False):
        cfg.MODEL.LORA.ENABLED = False
    network = build_salttrack(cfg, training=False)
    if checkpoint_override:
        checkpoint = Path(checkpoint_override)
    else:
        checkpoint = Path(env.save_dir) / "checkpoints" / "train" / "salttrack" / yaml_name / f"SALTTrack_ep{epoch:04d}.pth.tar"
    state = torch.load(str(checkpoint), map_location="cpu")
    state_dict = state["net"] if isinstance(state, dict) and "net" in state else state
    missing, unexpected = network.load_state_dict(state_dict, strict=False)
    if missing:
        print("Missing keys:", missing)
    if unexpected:
        print("Unexpected keys:", unexpected)
    network.to(device).eval()
    return network, checkpoint


def initialize_state(network, preprocessor, image, init_bbox, init_nlp, device):
    z_patch, resize_factor = sample_target(image, init_bbox, cfg.TEST.TEMPLATE_FACTOR, output_sz=cfg.TEST.TEMPLATE_SIZE)
    template = preprocessor.process(z_patch)
    template_list = [template] * cfg.TEST.NUM_TEMPLATES
    bbox_mask = build_template_mask(init_bbox, resize_factor, cfg.TEST.TEMPLATE_SIZE, device)
    soft_token_template_mask = [bbox_mask, bbox_mask]
    text_features, text_subject_features, _, _ = network.forward_text([init_nlp], 1, None, device=template.device)
    return TrackerState(
        template_list=template_list,
        soft_token_template_mask=soft_token_template_mask,
        text_features=text_features,
        text_subject_features=text_subject_features,
        temporal_infor=[],
        state=list(init_bbox),
        first_frame_flag=True,
    )


def map_box_back(pred_box, prev_state, search_size, resize_factor):
    cx_prev = prev_state[0] + 0.5 * prev_state[2]
    cy_prev = prev_state[1] + 0.5 * prev_state[3]
    cx, cy, w, h = pred_box
    half_side = 0.5 * search_size / resize_factor
    cx_real = cx + (cx_prev - half_side)
    cy_real = cy + (cy_prev - half_side)
    return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]


def normalize_score_map(score_map):
    score = score_map.detach().float().cpu().squeeze().numpy()
    score = score - score.min()
    denom = score.max()
    if denom > 1e-8:
        score = score / denom
    return score


def overlay_score_map(search_patch, score_map, pred_box_crop, gt_box_crop, output_path, title):
    score = normalize_score_map(score_map)
    heat = cv2.resize(score, (search_patch.shape[1], search_patch.shape[0]))
    heat = np.uint8(np.clip(heat * 255.0, 0, 255))
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(np.clip(0.58 * search_patch + 0.42 * heat, 0, 255))

    canvas = overlay.copy()
    for box, color in [(gt_box_crop, (0, 255, 0)), (pred_box_crop, (255, 60, 60))]:
        if box is None:
            continue
        x, y, w, h = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
    cv2.putText(canvas, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def route_rows(model, yaml_name, epoch, seq_name, frame_id):
    rows = []
    for layer_name, weights in collect_lora_router_weights_named(model):
        weight = weights.detach().float().cpu().reshape(-1)
        route_info = parse_route_layer(layer_name)
        rows.append({
            "model": yaml_name,
            "epoch": epoch,
            "sequence": seq_name,
            "frame": frame_id,
            "layer": layer_name,
            "route_module": route_info["module"],
            "route_depth": route_info["depth"],
            "route_sublayer": route_info["sublayer"],
            "route_label": route_info["label"],
            "semantic_route_mean": float(weight.mean().item()),
            "semantic_route_std": float(weight.std(unbiased=False).item()) if weight.numel() > 1 else 0.0,
        })
    return rows


def parse_route_layer(layer_name):
    """Convert raw module path into paper-friendly route metadata."""
    if layer_name.startswith("vl_fusion."):
        match = re.search(r"VLFusion_layers\.(\d+)\.([^.]+)$", layer_name)
        depth = int(match.group(1)) + 1 if match else -1
        sublayer = match.group(2).upper() if match else layer_name.split(".")[-1]
        return {
            "module": "VL Fusion",
            "depth": depth,
            "sublayer": sublayer,
            "label": f"VL-{depth} {sublayer}",
        }
    if layer_name.startswith("visual_temporal_fusion."):
        match = re.search(r"decoder\.layers\.(\d+)\.([^.]+)$", layer_name)
        depth = int(match.group(1)) + 1 if match else -1
        sublayer = match.group(2).replace("linear", "FFN-") if match else layer_name.split(".")[-1]
        return {
            "module": "Temporal Fusion",
            "depth": depth,
            "sublayer": sublayer,
            "label": f"Temp-{depth} {sublayer}",
        }
    if layer_name.startswith("language_adjust."):
        match = re.search(r"decoder\.layers\.(\d+)\.([^.]+)$", layer_name)
        depth = int(match.group(1)) + 1 if match else -1
        sublayer = match.group(2).replace("linear", "FFN-") if match else layer_name.split(".")[-1]
        return {
            "module": "Language Adjust",
            "depth": depth,
            "sublayer": sublayer,
            "label": f"Lang-{depth} {sublayer}",
        }
    if layer_name.startswith("confidence_pred."):
        sublayer = layer_name.split(".")[-1].upper()
        return {
            "module": "Confidence Head",
            "depth": 1,
            "sublayer": sublayer,
            "label": f"Conf {sublayer}",
        }
    return {
        "module": "Other",
        "depth": -1,
        "sublayer": layer_name.split(".")[-1],
        "label": layer_name[-32:],
    }


def run_one_model(args, yaml_name, epoch, model_out_dir, display_name):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device) if device.type == "cuda" else None
    load_model.disable_lora = args.disable_lora
    model, checkpoint = load_model(yaml_name, epoch, device, args.checkpoint)
    epoch_text = f"ep{epoch:04d}" if epoch >= 0 else "external"
    print(f"Loaded {display_name} ({yaml_name}, {epoch_text}): {checkpoint}")

    sequence_list = load_sequence_list(args.dataset)
    preprocessor = Preprocessor()
    feat_sz = cfg.TEST.SEARCH_SIZE // cfg.MODEL.BACKBONE.STRIDE
    output_window = None
    if cfg.TEST.WINDOW:
        from lib.test.utils.hann import hann2d

        output_window = hann2d(torch.tensor([feat_sz, feat_sz]).long(), centered=True).to(device)

    heatmap_dir = model_out_dir / "score_heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    semantic_rows = []
    route_data = []
    pred_features = []
    gt_features = []
    text_features = []
    heatmap_count = 0

    for seq_idx, seq in enumerate(sequence_list[: args.max_seq]):
        init_image = read_rgb(seq.frames[0])
        init_bbox = seq.init_info()["init_bbox"]
        init_nlp = seq.init_info().get("init_nlp", "")
        state = initialize_state(model, preprocessor, init_image, init_bbox, init_nlp, device)
        frame_ids = list(range(1, len(seq.frames), max(1, args.frame_stride)))[: args.frames_per_seq]

        for frame_id in frame_ids:
            image = read_rgb(seq.frames[frame_id])
            height, width, _ = image.shape
            prev_state = list(state.state)
            x_patch, resize_factor = sample_target(image, prev_state, cfg.TEST.SEARCH_FACTOR, output_sz=cfg.TEST.SEARCH_SIZE)
            search = preprocessor.process(x_patch)

            gt_xywh = torch.tensor(seq.ground_truth_rect[frame_id], dtype=torch.float32, device=device)
            crop_gt_xywh = transform_image_to_crop(
                gt_xywh,
                torch.tensor(prev_state, dtype=torch.float32, device=device),
                resize_factor,
                torch.tensor([cfg.TEST.SEARCH_SIZE, cfg.TEST.SEARCH_SIZE], dtype=torch.float32, device=device),
                normalize=True,
            ).clamp(min=0.0, max=1.0).view(1, 4)

            with torch.no_grad():
                out = model(
                    state.template_list,
                    search,
                    state.soft_token_template_mask,
                    exp_str=state.text_features,
                    exp_subject_mask=state.text_subject_features,
                    search_anno=crop_gt_xywh,
                    temporal_infor=state.temporal_infor,
                    first_frame_flag=state.first_frame_flag,
                    training=False,
                    return_visualization=True,
                )

            state.first_frame_flag = False
            state.temporal_infor = out["temporal_infor"]

            pred_score_map = out["score_map"]
            response = output_window * pred_score_map if output_window is not None else pred_score_map
            pred_boxes, best_score = model.box_head.cal_bbox(response, out["size_map"], out["offset_map"], return_score=True)
            pred_box_norm = pred_boxes.view(-1, 4).mean(dim=0)
            pred_box_crop_cxcywh = (pred_box_norm * cfg.TEST.SEARCH_SIZE).detach().cpu().tolist()
            pred_box_crop = [
                pred_box_crop_cxcywh[0] - 0.5 * pred_box_crop_cxcywh[2],
                pred_box_crop_cxcywh[1] - 0.5 * pred_box_crop_cxcywh[3],
                pred_box_crop_cxcywh[2],
                pred_box_crop_cxcywh[3],
            ]
            pred_box_global = (pred_box_norm * cfg.TEST.SEARCH_SIZE / resize_factor).detach().cpu().tolist()
            state.state = clip_box(map_box_back(pred_box_global, prev_state, cfg.TEST.SEARCH_SIZE, resize_factor), height, width, margin=10)

            pred_feat = out.get("pred_visual_features")
            gt_feat = out.get("gt_visual_features")
            text_feat = out.get("target_text_features")
            pred_text_sim = float(F.cosine_similarity(pred_feat, text_feat).mean().item())
            gt_text_sim = float(F.cosine_similarity(gt_feat, text_feat).mean().item()) if gt_feat is not None else np.nan
            pred_gt_sim = float(F.cosine_similarity(pred_feat, gt_feat).mean().item()) if gt_feat is not None else np.nan

            semantic_rows.append({
                "model": display_name,
                "yaml": yaml_name,
                "epoch": epoch,
                "sequence": seq.name,
                "frame": frame_id,
                "pred_text_cos": pred_text_sim,
                "gt_text_cos": gt_text_sim,
                "pred_gt_cos": pred_gt_sim,
                "semantic_distance": 1.0 - pred_text_sim,
                "best_score": float(best_score.view(-1)[0].item()),
            })
            route_data.extend(route_rows(model, display_name, epoch, seq.name, frame_id))
            pred_features.append(pred_feat.detach().cpu().squeeze(0).numpy())
            gt_features.append(gt_feat.detach().cpu().squeeze(0).numpy())
            text_features.append(text_feat.detach().cpu().squeeze(0).numpy())

            if heatmap_count < args.heatmaps:
                gt_box_crop = (crop_gt_xywh.squeeze(0) * cfg.TEST.SEARCH_SIZE).detach().cpu().tolist()
                overlay_score_map(
                    x_patch,
                    pred_score_map,
                    pred_box_crop,
                    gt_box_crop,
                    heatmap_dir / f"{seq_idx:03d}_{seq.name}_f{frame_id:04d}.jpg",
                    f"{display_name} {epoch_text} s={best_score.view(-1)[0].item():.3f}",
                )
                heatmap_count += 1

    write_csv(model_out_dir / "semantic_samples.csv", semantic_rows)
    write_csv(model_out_dir / "route_samples.csv", route_data)
    np.savez_compressed(
        model_out_dir / "semantic_features.npz",
        pred=np.asarray(pred_features),
        gt=np.asarray(gt_features),
        text=np.asarray(text_features),
    )
    plot_similarity(model_out_dir / "similarity_distribution.png", semantic_rows, display_name)
    plot_routes(model_out_dir / "route_by_layer.png", route_data, display_name)
    plot_routes_by_depth(model_out_dir / "route_by_depth.png", route_data, display_name)
    plot_scatter(model_out_dir / "semantic_scatter_pca.png", pred_features, gt_features, text_features, display_name)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_similarity(path, rows, title):
    if not rows:
        return
    values = {
        "pred-text": [r["pred_text_cos"] for r in rows],
        "gt-text": [r["gt_text_cos"] for r in rows],
        "pred-gt": [r["pred_gt_cos"] for r in rows],
    }
    plt.figure(figsize=(7.0, 4.2))
    plt.boxplot([values[k] for k in values], tick_labels=list(values.keys()), showmeans=True)
    plt.ylabel("cosine similarity")
    plt.title(title)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def plot_routes(path, rows, title):
    if not rows:
        return
    grouped: Dict[Tuple[str, int, str, str], List[float]] = {}
    for row in rows:
        info = route_info_from_row(row)
        key = (info["module"], info["depth"], info["sublayer"], info["label"])
        grouped.setdefault(key, []).append(float(row["semantic_route_mean"]))

    module_order = {
        "Language Adjust": 0,
        "VL Fusion": 1,
        "Temporal Fusion": 2,
        "Confidence Head": 3,
        "Other": 4,
    }
    sublayer_order = {
        "Q": 0,
        "K": 1,
        "V": 2,
        "PROJ": 3,
        "self_attn": 4,
        "multihead_attn": 5,
        "FFN-1": 6,
        "FFN-2": 7,
        "FC1": 8,
        "FC2": 9,
    }
    items = sorted(
        [(key, float(np.mean(values))) for key, values in grouped.items()],
        key=lambda item: (
            module_order.get(item[0][0], 99),
            item[0][1],
            sublayer_order.get(item[0][2], 99),
            item[0][2],
        ),
    )
    labels = [item[0][3] for item in items]
    means = [item[1] for item in items]
    colors = [module_color(item[0][0]) for item in items]

    plt.figure(figsize=(8.2, max(4.2, 0.24 * len(labels))))
    plt.barh(np.arange(len(labels)), means, color=colors)
    plt.yticks(np.arange(len(labels)), labels, fontsize=8)
    plt.xlabel("semantic expert route probability")
    plt.title(title)
    plt.xlim(0.0, 1.0)
    plt.grid(axis="x", alpha=0.25)
    plt.axvline(0.5, color="#555555", linewidth=1.0, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def route_info_from_row(row):
    if "route_module" in row and row["route_module"]:
        return {
            "module": row["route_module"],
            "depth": int(row["route_depth"]),
            "sublayer": row["route_sublayer"],
            "label": row["route_label"],
        }
    return parse_route_layer(row["layer"])


def module_color(module):
    return {
        "Language Adjust": "#6c63b7",
        "VL Fusion": "#2f7ebc",
        "Temporal Fusion": "#2f9b6d",
        "Confidence Head": "#c77c2f",
    }.get(module, "#777777")


def plot_routes_by_depth(path, rows, title):
    if not rows:
        return
    grouped: Dict[Tuple[str, int], List[float]] = {}
    for row in rows:
        info = route_info_from_row(row)
        if info["depth"] < 0:
            continue
        grouped.setdefault((info["module"], info["depth"]), []).append(float(row["semantic_route_mean"]))

    module_order = ["Language Adjust", "VL Fusion", "Temporal Fusion", "Confidence Head"]
    plt.figure(figsize=(6.6, 4.2))
    for module in module_order:
        items = sorted((depth, values) for (mod, depth), values in grouped.items() if mod == module)
        if not items:
            continue
        xs = [depth for depth, _ in items]
        ys = [float(np.mean(values)) for _, values in items]
        plt.plot(xs, ys, marker="o", linewidth=2.0, label=module, color=module_color(module))
    plt.axhline(0.5, color="#555555", linewidth=1.0, linestyle="--", alpha=0.6)
    plt.ylim(0.0, 1.0)
    plt.xlabel("module depth")
    plt.ylabel("semantic expert route probability")
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=240)
    plt.close()


def plot_scatter(path, pred_features, gt_features, text_features, title):
    if not pred_features:
        return
    features = np.concatenate([pred_features, gt_features, text_features], axis=0)
    labels = (
        ["pred"] * len(pred_features)
        + ["gt"] * len(gt_features)
        + ["text"] * len(text_features)
    )
    features = features - features.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(features, full_matrices=False)
    xy = features @ vh[:2].T
    colors = {"pred": "#d64f4f", "gt": "#2c9c69", "text": "#3f66c2"}
    plt.figure(figsize=(5.5, 5.0))
    for label in ["pred", "gt", "text"]:
        idx = np.array([l == label for l in labels])
        plt.scatter(xy[idx, 0], xy[idx, 1], s=18, alpha=0.72, label=label, c=colors[label])
    plt.legend(frameon=False)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main():
    args = parse_args()
    epoch = resolve_epoch(args)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    display_name = args.display_name or args.yaml
    suffix = f"ep{epoch:04d}" if epoch >= 0 else "external"
    run_one_model(args, args.yaml, epoch, out_root / f"{display_name}_{suffix}", display_name)


if __name__ == "__main__":
    main()
