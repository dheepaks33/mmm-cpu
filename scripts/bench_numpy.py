#!/usr/bin/env python3
"""Numpy/BLAS baseline: the article's "target" row (8 ms with MKL on an i7-6700).

Same method as bench.cpp: float32 inputs, warm-up, repeat until --min-time,
median wall-clock, GFLOP/s = 2*N^3 / median. Appends one row to --csv.

    python scripts/bench_numpy.py
    python scripts/bench_numpy.py --threads 1      # single-core BLAS
"""
import argparse
import csv
import os
import socket
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmmlib import cpu_model, disable_power_throttling, get_affinity_mask  # noqa: E402

try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:  # optional
    threadpool_info = None
    threadpool_limits = None

HEADER = ["timestamp_utc", "label", "host", "cpu", "compiler", "flags", "n", "kernel", "tile", "block",
          "threads", "reps", "min_ms", "median_ms", "mean_ms", "stdev_ms", "gflops", "rel_err", "status",
          "affinity"]


def blas_description():
    if threadpool_info is None:
        return "numpy %s (pip install threadpoolctl to see BLAS)" % np.__version__, -1
    for lib in threadpool_info():
        if lib.get("user_api") == "blas":
            return ("numpy %s / %s %s %s" % (np.__version__, lib.get("internal_api", "blas"),
                                              lib.get("version") or "?", lib.get("architecture") or ""),
                    lib.get("num_threads", -1))
    return "numpy %s / unknown BLAS" % np.__version__, -1


def measure(x, y, min_reps, min_time):
    np.dot(x, y)  # warm-up: thread pool start, page faults
    times = []
    z = None
    while len(times) < min_reps or sum(times) < min_time:
        t0 = time.perf_counter()
        z = np.dot(x, y)
        times.append(time.perf_counter() - t0)
        if len(times) >= 1000:
            break
    return times, z


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--min-time", type=float, default=2.0)
    p.add_argument("--min-reps", type=int, default=10)
    p.add_argument("--threads", type=int, default=None, help="limit BLAS threads")
    p.add_argument("--csv", default=None)
    p.add_argument("--label", default="numpy (BLAS)")
    args = p.parse_args()
    disable_power_throttling()

    n = args.n
    rng = np.random.default_rng(42)
    x = rng.standard_normal((n, n)).astype(np.float32)
    y = rng.standard_normal((n, n)).astype(np.float32)

    if args.threads and threadpool_limits is not None:
        with threadpool_limits(limits=args.threads, user_api="blas"):
            desc, threads = blas_description()
            times, z = measure(x, y, args.min_reps, args.min_time)
    else:
        desc, threads = blas_description()
        times, z = measure(x, y, args.min_reps, args.min_time)

    ref = x.astype(np.float64) @ y.astype(np.float64)  # sanity check vs float64
    rel = float(np.max(np.abs(z - ref)) / np.max(np.abs(ref)))
    med = statistics.median(times)
    gflops = 2.0 * n ** 3 / med / 1e9
    stdev = statistics.stdev(times) if len(times) > 1 else 0.0
    print("%-34s N=%d threads=%-2s reps=%-3d  median %9.2f ms  min %9.2f ms  stdev %6.2f ms  %7.2f GFLOP/s  [%s]"
          % ("numpy.dot", n, threads, len(times), med * 1e3, min(times) * 1e3, stdev * 1e3, gflops, desc),
          flush=True)

    if args.csv:
        fresh = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f, lineterminator="\n")
            if fresh:
                w.writerow(HEADER)
            w.writerow([datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), args.label,
                        socket.gethostname(), cpu_model(), desc, "", n, "numpy", "", "", threads,
                        len(times), "%.4f" % (min(times) * 1e3), "%.4f" % (med * 1e3),
                        "%.4f" % (statistics.mean(times) * 1e3), "%.4f" % (stdev * 1e3),
                        "%.3f" % gflops, "%.2e" % rel, "ok", get_affinity_mask()])


if __name__ == "__main__":
    main()
