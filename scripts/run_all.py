#!/usr/bin/env python3
"""Build, run every step of the article, run the numpy baseline, verify against
numpy, and write results/<timestamp>/{results.csv, summary.md, chart.png,
system.txt, build.log}. Works the same in PowerShell, cmd, bash and zsh.

    python scripts/run_all.py                          # full run, 1-3 minutes
    python scripts/run_all.py --quick                  # skip the slow -O0 step
    python scripts/run_all.py --min-time 5             # longer measurement, less noise
    python scripts/run_all.py --tile 8 --block 128     # your tuned sizes
    python scripts/run_all.py --affinity 0xFFF --threads 6   # pin to P-cores (see: mmm topology)
    python scripts/run_all.py --ghz 3.1 --cores 6      # add FLOPs/core/cycle to the summary
    python scripts/run_all.py --n 2048                 # bigger matrices (article column then blank)
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmmlib import (ROOT, cpus_in_mask, disable_power_throttling, exe,  # noqa: E402
                    get_affinity_mask, logical_cpus, set_affinity)

SCRIPTS = ROOT / "scripts"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="skip step 1 (naive at -O0, slow)")
    ap.add_argument("--min-time", type=float, default=2.0, help="seconds measured per step")
    ap.add_argument("--threads", type=int, help="threads for step 6 and numpy")
    ap.add_argument("--tile", type=int, default=16, help="inner tile size for steps 5-6")
    ap.add_argument("--block", type=int, default=256, help="row/col block size for step 6")
    ap.add_argument("--affinity", help="hex CPU mask to pin every run to, e.g. 0xFFF")
    ap.add_argument("--ghz", type=float, help="sustained all-core clock, for FLOPs/core/cycle")
    ap.add_argument("--cores", type=int, help="physical cores used, for FLOPs/core/cycle")
    ap.add_argument("--n", type=int, default=1024, help="matrix size, multiple of 256")
    ap.add_argument("--no-build", action="store_true", help="use the binaries already in build/")
    a = ap.parse_args()

    if a.affinity:
        mask = int(a.affinity, 0)
        set_affinity(mask)  # inherited by every program started below
        default_threads = cpus_in_mask(mask)
    else:
        default_threads = logical_cpus()
    # Deliberately ignores a leftover OMP_NUM_THREADS from manual runs: use --threads.
    threads = a.threads or default_threads
    os.environ["OMP_NUM_THREADS"] = str(threads)
    disable_power_throttling()

    out = ROOT / "results" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)
    csv_path = out / "results.csv"
    py = sys.executable
    print("==> results go to %s" % out.relative_to(ROOT))
    print("==> threads=%d  affinity=%s  tile=%d  block=%d" % (threads, get_affinity_mask(), a.tile, a.block))

    if not a.no_build:
        print("==> building")
        with open(out / "build.log", "w", encoding="utf-8") as log:
            r = subprocess.run([py, str(SCRIPTS / "build.py"), "--n", str(a.n)], stdout=log,
                               stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print((out / "build.log").read_text(encoding="utf-8"))
            return 1
    with open(out / "system.txt", "w", encoding="utf-8") as f:
        subprocess.run([py, str(SCRIPTS / "sysinfo.py")], stdout=f, stderr=subprocess.STDOUT)

    failures = []

    def step(label, binary, args):
        print("%-28s " % label, end="", flush=True)
        r = subprocess.run([str(exe(binary))] + args + ["--csv", str(csv_path), "--label", label,
                                                         "--min-time", str(a.min_time)])
        if r.returncode != 0:
            failures.append(label)
            print("   ^ exit code %d" % r.returncode)
        time.sleep(2)  # let the CPU cool a little between steps

    if not a.quick:
        step("1 naive (-O0)", "mmm_O0", ["naive", "--min-reps", "3"])
    step("2 naive + flags", "mmm", ["naive", "--min-reps", "3"])
    step("3 + register acc", "mmm", ["regacc", "--min-reps", "3"])
    step("4 loop reorder (RIC)", "mmm", ["reorder"])
    step("5 + L1 tiling (%d)" % a.tile, "mmm", ["tiled", "--tile", str(a.tile)])
    dump = out / "dump"
    step("6 + multithreading", "mmm", ["parallel", "--tile", str(a.tile), "--block", str(a.block),
                                       "--dump", str(dump)])

    print("%-28s " % "7 numpy (BLAS)", end="", flush=True)
    r = subprocess.run([py, str(SCRIPTS / "bench_numpy.py"), "--csv", str(csv_path), "--label",
                        "7 numpy (BLAS)", "--min-time", str(a.min_time), "--threads", str(threads),
                        "--n", str(a.n)])
    if r.returncode != 0:
        failures.append("7 numpy")

    if dump.exists():
        print("==> cross-checking step 6 output against numpy")
        r = subprocess.run([py, str(SCRIPTS / "verify_numpy.py"), str(dump)])
        if r.returncode == 0:
            shutil.rmtree(dump)
        else:
            failures.append("verify vs numpy")

    print("==> summary")
    plot = [py, str(SCRIPTS / "plot.py"), str(out)]
    if a.ghz:
        plot += ["--ghz", str(a.ghz), "--cores", str(a.cores or threads)]
    subprocess.run(plot)
    rel = out.relative_to(ROOT)
    print("==> done: %s" % ", ".join(str(rel / f) for f in
                                     ("summary.md", "chart.png", "results.csv", "system.txt")))
    if failures:
        print("==> FAILED: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
