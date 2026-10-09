#!/usr/bin/env python3
"""
Measure LaneNet forward-pass latency on CPU and save it with the hardware info.

This times the network only (random weights, random 1x3xHxW input, no
pre/post-processing). Latency does not depend on whether the weights are
trained, but it says nothing about accuracy.

    python scripts/benchmark_cpu.py --threads 1 --output results/cpu_latency.json
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import create_lanenet  # noqa: E402


def cpu_name() -> str:
    """Best-effort human-readable CPU name."""
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def benchmark(backbone: str, height: int, width: int, warmup: int, runs: int) -> dict:
    model = create_lanenet(backbone=backbone, pretrained=False).eval()
    x = torch.randn(1, 3, height, width)
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        for _ in range(runs):
            start = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - start) * 1000)
    times = np.array(times)
    return {
        "backbone": backbone,
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "median_ms": round(float(np.median(times)), 1),
        "p90_ms": round(float(np.percentile(times, 90)), 1),
        "fps_at_median": round(1000 / float(np.median(times)), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--output", type=str, default=None, help="Write results as JSON")
    args = parser.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)

    results = {
        "hardware": {
            "cpu": cpu_name(),
            "os": f"{platform.system()} {platform.release()}",
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "input": f"1x3x{args.height}x{args.width}",
        "runs": args.runs,
        "models": [benchmark(b, args.height, args.width, args.warmup, args.runs)
                   for b in ("efficientnet", "mobilenet")],
    }
    print(json.dumps(results, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
