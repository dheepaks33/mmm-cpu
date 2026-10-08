#!/usr/bin/env python3
"""The article's grid search over tile sizes.

Runs the tiled kernel (1 thread) and the parallel kernel (all threads) for
every tile size and writes results/sweep-<stamp>/{sweep.csv, sweep.png, system.txt}.

    python scripts/sweep_tiles.py
    python scripts/sweep_tiles.py --block 128 --threads 6 --affinity 0xFFF
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmmlib import ROOT, cpus_in_mask, disable_power_throttling, exe, logical_cpus, set_affinity  # noqa: E402

TILES = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--min-time", type=float, default=1.0)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--affinity", help="hex CPU mask, e.g. 0xFFF")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()

    default_threads = logical_cpus()
    if a.affinity:
        mask = int(a.affinity, 0)
        set_affinity(mask)
        default_threads = cpus_in_mask(mask)
    os.environ["OMP_NUM_THREADS"] = str(a.threads or default_threads)
    disable_power_throttling()

    py = sys.executable
    out = ROOT / "results" / ("sweep-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True)
    csv_path = out / "sweep.csv"
    if not a.no_build:
        with open(out / "build.log", "w", encoding="utf-8") as log:
            if subprocess.run([py, str(ROOT / "scripts" / "build.py")], stdout=log,
                              stderr=subprocess.STDOUT).returncode != 0:
                print((out / "build.log").read_text(encoding="utf-8"))
                return 1
    with open(out / "system.txt", "w", encoding="utf-8") as f:
        subprocess.run([py, str(ROOT / "scripts" / "sysinfo.py")], stdout=f, stderr=subprocess.STDOUT)

    for kernel in ("tiled", "parallel"):
        for t in TILES:
            args = [str(exe("mmm")), kernel, "--tile", str(t), "--csv", str(csv_path),
                    "--label", "%s %d" % (kernel, t), "--min-time", str(a.min_time)]
            if kernel == "parallel":
                args += ["--block", str(a.block)]
            subprocess.run(args)
    subprocess.run([py, str(ROOT / "scripts" / "plot.py"), str(out), "--sweep"])
    print("==> done: %s" % (out.relative_to(ROOT) / "sweep.png"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
