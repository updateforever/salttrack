#!/usr/bin/env python3

import csv
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml


ROOT = Path("/data/wyp/SALTTrack")
PYTHON_BIN = "/data/envs/wyp_vlt/bin/python"
SAVE_DIR = ROOT / "output"
CONFIG_DIR = ROOT / "experiments" / "salttrack"
BENCH_DIR = ROOT / "output" / "bench_lora_bs_4gpu"
LOG_DIR = BENCH_DIR / "logs"
SUMMARY_CSV = BENCH_DIR / "summary.csv"

CUDA_DEVICES = "0,1,2,3"
NUM_GPUS = 4
PRINT_INTERVAL = 10
MIN_SAMPLE_PER_EPOCH = 640
EPOCH = 1
NUM_WORKER = 4
MAX_WAIT_SEC = 1200
POLL_SEC = 2
TARGET_MEM_MIB = 22500

TESTS = [
    ("base_lora", "salttrack_base", 16, 256),
    ("base_backbone_lora", "salttrack_base_backbone_lora", 4, 128),
    ("large_lora", "salttrack_large", 8, 128),
    ("large_backbone_lora", "salttrack_large_backbone_lora", 2, 96),
]


def run(cmd, **kwargs):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **kwargs)


def gpu_memory_used():
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
        "-i",
        CUDA_DEVICES,
    ]
    result = run(cmd)
    if result.returncode != 0:
        return []
    return [int(x.strip()) for x in result.stdout.splitlines() if x.strip()]


def create_config(src_name, bench_name, batch_size):
    src = CONFIG_DIR / f"{src_name}.yaml"
    dst = CONFIG_DIR / f"{bench_name}.yaml"
    with src.open("r") as f:
        cfg = yaml.safe_load(f)

    cfg["DATA"]["TRAIN"]["SAMPLE_PER_EPOCH"] = max(
        MIN_SAMPLE_PER_EPOCH, batch_size * NUM_GPUS * PRINT_INTERVAL * 3
    )
    cfg["TRAIN"]["EPOCH"] = EPOCH
    cfg["TRAIN"]["BATCH_SIZE"] = batch_size
    cfg["TRAIN"]["NUM_WORKER"] = NUM_WORKER
    cfg["TRAIN"]["PRINT_INTERVAL"] = PRINT_INTERVAL
    cfg["TRAIN"]["SAVE_INTERVAL"] = 1000

    with dst.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return dst


def cleanup_config(path):
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def cleanup_outputs(bench_name):
    for path in [
        SAVE_DIR / "logs" / f"salttrack-{bench_name}.log",
        SAVE_DIR / "tensorboard" / "train" / "salttrack" / bench_name,
        SAVE_DIR / "checkpoints" / "train" / "salttrack" / bench_name,
    ]:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def parse_log(log_path):
    fps_values = []
    batch_fps_values = []
    if not log_path.exists():
        return None, None

    pattern = re.compile(r"FPS:\s*([0-9.]+)\s*\(([0-9.]+)\)")
    for line in log_path.read_text(errors="ignore").splitlines():
        match = pattern.search(line)
        if match:
            fps_values.append(float(match.group(1)))
            batch_fps_values.append(float(match.group(2)))

    if not fps_values:
        return None, None
    return fps_values[-1], batch_fps_values[-1]


def parse_any_log(*log_paths):
    for log_path in log_paths:
        fps, batch_fps = parse_log(log_path)
        if fps is not None:
            return fps, batch_fps
    return None, None


def tail_contains_oom(log_path):
    if not log_path.exists():
        return False
    text = log_path.read_text(errors="ignore")[-20000:]
    return "cuda out of memory" in text.lower() or "out of memory" in text.lower()


def terminate_process(proc):
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except Exception:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def run_one(case_name, src_name, batch_size):
    bench_name = f"bench_{case_name}_bs{batch_size}"
    config_path = create_config(src_name, bench_name, batch_size)
    cleanup_outputs(bench_name)

    train_log = LOG_DIR / f"{bench_name}.stdout.log"
    trainer_log = SAVE_DIR / "logs" / f"salttrack-{bench_name}.log"
    before_mem = gpu_memory_used()
    peak_mem = before_mem[:]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICES
    env["OPENCV_OPENCL_RUNTIME"] = "disabled"
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["NUMEXPR_NUM_THREADS"] = "4"

    cmd = [
        PYTHON_BIN,
        "-m",
        "torch.distributed.launch",
        "--nproc_per_node",
        str(NUM_GPUS),
        "lib/train/run_training.py",
        "--script",
        "salttrack",
        "--config",
        bench_name,
        "--save_dir",
        str(SAVE_DIR),
        "--use_lmdb",
        "0",
        "--use_wandb",
        "0",
    ]

    status = "running"
    start = time.time()
    with train_log.open("w") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
            text=True,
        )

        while True:
            elapsed = time.time() - start
            mem = gpu_memory_used()
            if mem:
                peak_mem = [max(a, b) for a, b in zip(peak_mem or mem, mem)]

            code = proc.poll()
            if code is not None:
                status = "ok" if code == 0 else f"failed:{code}"
                break

            if tail_contains_oom(train_log):
                status = "oom"
                terminate_process(proc)
                break

            if elapsed > MAX_WAIT_SEC:
                status = "timeout"
                terminate_process(proc)
                break

            time.sleep(POLL_SEC)

    elapsed = time.time() - start
    fps, batch_fps = parse_any_log(trainer_log, train_log)

    if status.startswith("failed") or status == "timeout":
        if tail_contains_oom(train_log):
            status = "oom"

    cleanup_config(config_path)

    return {
        "case": case_name,
        "source_config": src_name,
        "bench_config": bench_name,
        "batch_size_per_gpu": batch_size,
        "global_batch_size": batch_size * NUM_GPUS,
        "status": status,
        "elapsed_sec": round(elapsed, 1),
        "fps": fps,
        "batch_fps": batch_fps,
        "peak_mem_mib": max(peak_mem) if peak_mem else None,
        "peak_mem_per_gpu_mib": " ".join(map(str, peak_mem)) if peak_mem else "",
        "stdout_log": str(train_log),
        "trainer_log": str(trainer_log),
    }


def append_row(row):
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    exists = SUMMARY_CSV.exists()
    with SUMMARY_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def is_success(row):
    return row["status"] == "ok" and row["fps"] is not None


def run_case(case_name, src_name, start_bs, max_bs):
    rows = []
    ok = None
    oom = None
    bs = start_bs

    while bs <= max_bs:
        print(f"[BENCH] {case_name} bs={bs}", flush=True)
        row = run_one(case_name, src_name, bs)
        rows.append(row)
        append_row(row)
        print(row, flush=True)

        if row["status"] == "oom":
            oom = bs
            break
        if not is_success(row):
            break

        ok = row
        if row["peak_mem_mib"] and row["peak_mem_mib"] >= TARGET_MEM_MIB:
            break
        bs *= 2

    if oom is not None and ok is not None:
        low = ok["batch_size_per_gpu"] + 1
        high = oom - 1
        while low <= high:
            mid = (low + high) // 2
            print(f"[BENCH] {case_name} bs={mid}", flush=True)
            row = run_one(case_name, src_name, mid)
            rows.append(row)
            append_row(row)
            print(row, flush=True)

            if row["status"] == "oom":
                high = mid - 1
            elif is_success(row):
                ok = row
                low = mid + 1
            else:
                break

    return rows


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if SUMMARY_CSV.exists():
        SUMMARY_CSV.unlink()

    rows = []
    for case_name, src_name, start_bs, max_bs in TESTS:
        rows.extend(run_case(case_name, src_name, start_bs, max_bs))

    print(f"[DONE] summary: {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
