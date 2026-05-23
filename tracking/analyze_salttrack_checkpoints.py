import argparse
import csv
import os
import re
from pathlib import Path

import torch

import _init_paths  # noqa: F401
from lib.test.analysis.extract_results import extract_results
from lib.test.analysis.plot_results import get_auc_curve, get_prec_curve
from lib.test.evaluation import get_dataset, trackerlist
from lib.test.evaluation.environment import env_settings


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze SALTTrack checkpoint inference results.")
    parser.add_argument("--tracker_name", type=str, default="salttrack")
    parser.add_argument("--tracker_param", type=str, default="salttrack_base")
    parser.add_argument("--dataset_name", type=str, default="tnl2k")
    parser.add_argument("--start_epoch", type=int, default=70)
    parser.add_argument("--output_csv", type=str, default="")
    return parser.parse_args()


def discover_run_tags(save_dir: Path, tracker_param: str, start_epoch: int):
    result_root = save_dir / tracker_param
    if not result_root.is_dir():
        raise FileNotFoundError(f"Result root not found: {result_root}")

    run_tags = []
    pattern = re.compile(rf"^{re.escape(tracker_param)}_ep(\d+)$")
    for path in result_root.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        epoch = int(match.group(1))
        if epoch >= start_epoch:
            run_tags.append((epoch, path.name))

    return [tag for _, tag in sorted(run_tags)]


def compute_metrics(tracker_name: str, tracker_param: str, dataset_name: str, run_tag: str, dataset):
    settings = env_settings()
    trackers = trackerlist(name=tracker_name,
                           parameter_name=tracker_param,
                           dataset_name=dataset_name,
                           run_ids=None,
                           display_name=run_tag)
    for tracker in trackers:
        tracker.results_dir = f"{settings.save_dir}/{tracker_param}/{run_tag}"

    report_name = f"{dataset_name}_{tracker_param}_{run_tag}"
    eval_data = extract_results(trackers, dataset, report_name)
    valid_sequence = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)

    overlap = torch.tensor(eval_data["ave_success_rate_plot_overlap"])
    center = torch.tensor(eval_data["ave_success_rate_plot_center"])
    center_norm = torch.tensor(eval_data["ave_success_rate_plot_center_norm"])

    _, auc = get_auc_curve(overlap, valid_sequence)
    _, precision = get_prec_curve(center, valid_sequence)
    _, norm_precision = get_prec_curve(center_norm, valid_sequence)

    return {
        "run_tag": run_tag,
        "auc": auc[0].item(),
        "precision": precision[0].item(),
        "norm_precision": norm_precision[0].item(),
        "valid_sequences": int(valid_sequence.long().sum().item()),
        "total_sequences": int(valid_sequence.shape[0]),
    }


def main():
    args = parse_args()
    settings = env_settings()
    save_dir = Path(settings.save_dir)

    run_tags = discover_run_tags(save_dir, args.tracker_param, args.start_epoch)
    if not run_tags:
        raise RuntimeError(f"No run tags found for {args.tracker_param} from epoch {args.start_epoch}")

    dataset = get_dataset(args.dataset_name)
    rows = []
    for run_tag in run_tags:
        print(f"Analyzing {run_tag} ...", flush=True)
        rows.append(compute_metrics(args.tracker_name, args.tracker_param, args.dataset_name, run_tag, dataset))

    rows = sorted(rows, key=lambda item: item["auc"], reverse=True)

    output_csv = args.output_csv
    if not output_csv:
        output_csv = str(save_dir / "eval_logs" / args.tracker_param /
                         f"{args.dataset_name}_metrics_from{args.start_epoch}.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "run_tag", "auc", "precision", "norm_precision", "valid_sequences", "total_sequences"
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print("")
    print("run_tag,AUC,P,NP,valid/total")
    for row in rows:
        print("{},{:.2f},{:.2f},{:.2f},{}/{}".format(
            row["run_tag"],
            row["auc"],
            row["precision"],
            row["norm_precision"],
            row["valid_sequences"],
            row["total_sequences"],
        ))
    print("")
    print(f"Saved metrics CSV: {output_csv}")


if __name__ == "__main__":
    main()
