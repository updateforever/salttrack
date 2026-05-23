#!/usr/bin/env python3
"""Replot paper-friendly routed-LoRA figures from an existing route_samples.csv."""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracking.visualization.export_salttrack_semantic_visualizations import plot_routes, plot_routes_by_depth


def parse_args():
    parser = argparse.ArgumentParser(description="Replot SALTTrack route figures from CSV.")
    parser.add_argument("route_csv", help="Path to route_samples.csv")
    parser.add_argument("--title", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    route_csv = Path(args.route_csv)
    with open(route_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    title = args.title or route_csv.parent.name
    plot_routes(route_csv.parent / "route_by_layer_pretty.png", rows, title)
    plot_routes_by_depth(route_csv.parent / "route_by_depth.png", rows, title)
    print("Saved:", route_csv.parent / "route_by_layer_pretty.png")
    print("Saved:", route_csv.parent / "route_by_depth.png")


if __name__ == "__main__":
    main()
