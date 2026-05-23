#!/usr/bin/env python3

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("/data/wyp/SALTTrack")
OUT_DIR = ROOT / "output" / "plots" / "salttrack_nobb_routed_both"

TRAIN_LOGS = {
    "base_both": ROOT / "output/logs/salttrack-salttrack_base_nobb_routed_both.log",
    "large_both": ROOT / "output/logs/salttrack-salttrack_large_nobb_routed_both.log",
}

METRIC_CSVS = {
    "base_both": ROOT / "output/eval_logs/nobb_routed_tnl2k_metrics/tnl2k_salttrack_base_nobb_routed_both_metrics_from70.csv",
    "large_both": ROOT / "output/eval_logs/nobb_routed_tnl2k_metrics/tnl2k_salttrack_large_nobb_routed_both_metrics_from70.csv",
}


LINE_RE = re.compile(r"\[train:\s*(\d+),\s*(\d+)\s*/\s*(\d+)\]\s*FPS:\s*([0-9.]+)\s*\(([0-9.]+)\)")
KV_RE = re.compile(r"([A-Za-z0-9_/\-]+):\s*(-?[0-9]+(?:\.[0-9]+)?)")
EPOCH_RE = re.compile(r"ep(\d+)$")


def parse_train_log(path: Path):
    rows = []
    if not path.is_file():
        return rows

    with path.open(errors="ignore") as f:
        for line in f:
            match = LINE_RE.search(line)
            if not match:
                continue
            epoch, iter_idx, iter_total, fps, batch_fps = match.groups()
            row = {
                "epoch": int(epoch),
                "iter": int(iter_idx),
                "iter_total": int(iter_total),
                "step": (int(epoch) - 1) * int(iter_total) + int(iter_idx),
                "FPS/avg": float(fps),
                "FPS/interval": float(batch_fps),
            }
            for key, value in KV_RE.findall(line):
                if key == "FPS":
                    continue
                row[key] = float(value)
            rows.append(row)
    return rows


def write_train_csv(name: str, rows):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    preferred = ["epoch", "iter", "iter_total", "step", "FPS/avg", "FPS/interval"]
    ordered = preferred + [key for key in keys if key not in preferred]
    out_path = OUT_DIR / f"{name}_train_log_parsed.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def epoch_average(rows, key: str):
    values = defaultdict(list)
    for row in rows:
        if key in row:
            values[row["epoch"]].append(row[key])
    epochs = sorted(values)
    averaged = [sum(values[epoch]) / len(values[epoch]) for epoch in epochs]
    return epochs, averaged


def plot_train_curve(all_rows, keys, filename, ylabel=None):
    plt.figure(figsize=(8, 5))
    plotted = False
    for name, rows in all_rows.items():
        for key in keys:
            epochs, values = epoch_average(rows, key)
            if not epochs:
                continue
            label = name if len(keys) == 1 else f"{name} {key}"
            plt.plot(epochs, values, linewidth=2, label=label)
            plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("Epoch")
    plt.ylabel(ylabel or "Value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200)
    plt.close()


def parse_metric_csv(path: Path):
    rows = []
    if not path.is_file():
        return rows
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            match = EPOCH_RE.search(row["run_tag"])
            if not match:
                continue
            rows.append({
                "epoch": int(match.group(1)),
                "auc": float(row["auc"]),
                "precision": float(row["precision"]),
                "norm_precision": float(row["norm_precision"]),
            })
    return sorted(rows, key=lambda item: item["epoch"])


def plot_metric_curve(metric_rows, key, filename, ylabel):
    plt.figure(figsize=(8, 5))
    plotted = False
    for name, rows in metric_rows.items():
        if not rows:
            continue
        plt.plot([row["epoch"] for row in rows], [row[key] for row in rows],
                 marker="o", linewidth=2, label=name)
        plotted = True
    if not plotted:
        plt.close()
        return
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=200)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = {name: parse_train_log(path) for name, path in TRAIN_LOGS.items()}
    for name, rows in all_rows.items():
        write_train_csv(name, rows)

    plot_train_curve(all_rows, ["Loss/total"], "train_total_loss.png", "Total Loss")
    plot_train_curve(all_rows, ["Loss/location"], "train_location_loss.png", "Location Loss")
    plot_train_curve(all_rows, ["IoU_main"], "train_iou_main.png", "IoU")
    plot_train_curve(all_rows, ["Loss/semantic"], "train_semantic_loss.png", "Semantic Loss")
    plot_train_curve(all_rows, ["semantic_distance_loss"], "train_semantic_distance_loss.png", "Semantic Distance Loss")
    plot_train_curve(all_rows, ["semantic_direction_loss"], "train_semantic_direction_loss.png", "Semantic Direction Loss")
    plot_train_curve(all_rows, ["semantic_direction_consistency"], "train_semantic_direction_consistency.png",
                     "Direction Consistency")
    plot_train_curve(all_rows, ["gt_text_similarity", "pred_text_similarity"], "train_text_similarity.png",
                     "Text-Visual Similarity")
    plot_train_curve(all_rows, ["Loss/lora_semantic_reg"], "train_lora_semantic_reg.png", "LoRA Semantic Reg")
    plot_train_curve(all_rows, ["Loss/lora_router"], "train_lora_router_loss.png", "LoRA Router Loss")
    plot_train_curve(all_rows, ["lora_semantic_route", "semantic_gate"], "train_route_and_gate.png", "Gate / Route")
    plot_train_curve(all_rows, ["FPS/avg"], "train_fps.png", "Average FPS")
    plot_train_curve(all_rows, ["Mem/max_alloc"], "train_max_alloc_memory.png", "Max Allocated Memory (GB)")

    metric_rows = {name: parse_metric_csv(path) for name, path in METRIC_CSVS.items()}
    plot_metric_curve(metric_rows, "auc", "tnl2k_auc_by_epoch.png", "TNL2K AUC")
    plot_metric_curve(metric_rows, "precision", "tnl2k_precision_by_epoch.png", "TNL2K Precision")
    plot_metric_curve(metric_rows, "norm_precision", "tnl2k_norm_precision_by_epoch.png", "TNL2K Norm Precision")

    print(f"Saved plots and parsed CSVs to: {OUT_DIR}")
    for path in sorted(OUT_DIR.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
