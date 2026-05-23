#!/usr/bin/env python3
"""Plot comparison figures from multiple SALTTrack visualization folders."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SALTTrack visualization CSVs.")
    parser.add_argument("--output", required=True)
    parser.add_argument("semantic_csv", nargs="+")
    return parser.parse_args()


def read_rows(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["_source"] = str(path)
                rows.append(row)
    return rows


def plot_metric_boxplot(rows, output_dir, metric, ylabel):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(float(row[metric]))
    names = list(grouped.keys())
    values = [grouped[name] for name in names]

    plt.figure(figsize=(max(6.5, 1.4 * len(names)), 4.2))
    plt.boxplot(values, tick_labels=names, showmeans=True)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / f"{metric}_comparison.png", dpi=240)
    plt.close()


def write_summary(rows, output_dir):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["model"], []).append(row)
    metrics = ["pred_text_cos", "gt_text_cos", "pred_gt_cos", "semantic_distance", "best_score"]
    with open(output_dir / "semantic_comparison_summary.csv", "w", newline="") as f:
        fieldnames = ["model", "num_samples"]
        for metric in metrics:
            fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_median"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, model_rows in grouped.items():
            out = {"model": model, "num_samples": len(model_rows)}
            for metric in metrics:
                values = np.asarray([float(row[metric]) for row in model_rows], dtype=np.float64)
                out[f"{metric}_mean"] = float(values.mean())
                out[f"{metric}_std"] = float(values.std())
                out[f"{metric}_median"] = float(np.median(values))
            writer.writerow(out)


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_rows([Path(p) for p in args.semantic_csv])
    write_summary(rows, output_dir)
    plot_metric_boxplot(rows, output_dir, "pred_text_cos", "pred-text cosine similarity")
    plot_metric_boxplot(rows, output_dir, "gt_text_cos", "gt-text cosine similarity")
    plot_metric_boxplot(rows, output_dir, "semantic_distance", "1 - pred-text cosine")
    print("Saved comparison figures to", output_dir)


if __name__ == "__main__":
    main()
