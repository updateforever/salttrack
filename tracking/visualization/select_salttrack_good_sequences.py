#!/usr/bin/env python3
"""Select TNL2K sequences where SALTTrack has strong per-sequence behavior."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lib.test.evaluation.tnl2kdataset import TNL2kDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Select good sequences for qualitative SALTTrack visualization.")
    parser.add_argument("--ours-root", required=True, help="Directory containing our TNL2K result txt files.")
    parser.add_argument("--compare-root", default=None, help="Optional baseline result directory.")
    parser.add_argument("--output-dir", default="output/visualizations/selected_sequences")
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--min-ours-iou", type=float, default=0.55)
    parser.add_argument("--min-gain", type=float, default=0.03)
    return parser.parse_args()


def xywh_iou(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(len(a), len(b))
    a = a[:n]
    b = b[:n]
    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iw = np.maximum(ix2 - ix1, 0.0)
    ih = np.maximum(iy2 - iy1, 0.0)
    inter = iw * ih
    area_a = np.maximum(a[:, 2], 0.0) * np.maximum(a[:, 3], 0.0)
    area_b = np.maximum(b[:, 2], 0.0) * np.maximum(b[:, 3], 0.0)
    union = np.maximum(area_a + area_b - inter, 1e-8)
    return inter / union


def load_pred(root, seq_name):
    path = Path(root) / f"{seq_name}.txt"
    if not path.is_file():
        return None
    try:
        data = np.loadtxt(path, delimiter="\t")
    except ValueError:
        data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :4]


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = TNL2kDataset().get_sequence_list()
    rows = []
    for seq in dataset:
        ours = load_pred(args.ours_root, seq.name)
        if ours is None:
            continue
        gt = np.asarray(seq.ground_truth_rect, dtype=np.float64)
        ours_iou = xywh_iou(ours, gt)
        compare_iou = None
        if args.compare_root:
            compare = load_pred(args.compare_root, seq.name)
            if compare is not None:
                compare_iou = xywh_iou(compare, gt)
        ours_mean = float(np.mean(ours_iou))
        ours_success = float(np.mean(ours_iou > 0.5))
        if compare_iou is None:
            compare_mean = 0.0
            compare_success = 0.0
            gain = ours_mean
        else:
            n = min(len(ours_iou), len(compare_iou))
            ours_mean = float(np.mean(ours_iou[:n]))
            ours_success = float(np.mean(ours_iou[:n] > 0.5))
            compare_mean = float(np.mean(compare_iou[:n]))
            compare_success = float(np.mean(compare_iou[:n] > 0.5))
            gain = ours_mean - compare_mean
        rows.append({
            "sequence": seq.name,
            "length": len(seq.frames),
            "ours_mean_iou": ours_mean,
            "ours_success_0.5": ours_success,
            "compare_mean_iou": compare_mean,
            "compare_success_0.5": compare_success,
            "gain_mean_iou": gain,
        })

    filtered = [
        row for row in rows
        if row["ours_mean_iou"] >= args.min_ours_iou and row["gain_mean_iou"] >= args.min_gain
    ]
    if len(filtered) < args.topk:
        filtered = rows
    filtered = sorted(
        filtered,
        key=lambda row: (row["gain_mean_iou"], row["ours_mean_iou"], row["ours_success_0.5"]),
        reverse=True,
    )
    selected = filtered[: args.topk]

    with open(out_dir / "selected_sequences.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(selected)
    with open(out_dir / "selected_sequences.txt", "w") as f:
        for row in selected:
            f.write(row["sequence"] + "\n")

    print("Selected sequences:")
    for row in selected:
        print("{sequence}: ours IoU={ours_mean_iou:.3f}, compare IoU={compare_mean_iou:.3f}, gain={gain_mean_iou:.3f}".format(**row))
    print("Saved:", out_dir / "selected_sequences.csv")
    print("Saved:", out_dir / "selected_sequences.txt")


if __name__ == "__main__":
    main()
