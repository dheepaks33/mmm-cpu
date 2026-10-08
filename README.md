# mmm-cpu: replicate "Fast Multidimensional Matrix Multiplication on CPU from Scratch"

This code reproduces every step of Simon Boehm's article
(https://siboehm.com/articles/22/Fast-MMM-on-CPU) on your own machine. A
benchmark harness records everything you need to compare runs. The same
commands work in Windows PowerShell and in a Linux or macOS terminal.

## Quick start: Windows 11 (PowerShell)

```powershell
winget install -e --id MSYS2.MSYS2
winget install -e --id Python.Python.3.12
C:\msys64\usr\bin\bash.exe -lc "pacman -Syu --noconfirm"
C:\msys64\usr\bin\bash.exe -lc "pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-clang mingw-w64-ucrt-x86_64-llvm-openmp"
# open a new PowerShell window, cd into this folder, then:
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned   # once, lets PowerShell run Activate.ps1
.\.venv\Scripts\Activate.ps1
pip install numpy matplotlib threadpoolctl
python scripts\build.py
.\build\mmm.exe topology
python scripts\run_all.py
```

## Quick start: Ubuntu / WSL2 / macOS

```bash
sudo apt install -y build-essential clang libomp-dev python3-venv   # macOS: xcode-select --install && brew install libomp
python3 -m venv .venv && source .venv/bin/activate
pip install numpy matplotlib threadpoolctl
python scripts/build.py
python scripts/run_all.py
```

## What's here

| Path | Purpose |
|---|---|
| `src/kernels.h` | The 6 kernels from the article: naive, loop reorder, tiling, OpenMP |
| `src/bench.cpp` | Dependency-free harness: warm-up, repeats, median, GFLOP/s, correctness check, CSV. `mmm topology` shows core types, SIMD support and a P-core mask |
| `scripts/build.py` | Builds `build/mmm_O0` and `build/mmm` (adds `.exe` on Windows). `--asm`, `--vecreport`, `--n`, `--compiler` |
| `scripts/run_all.py` | One command: build, system info, 7 steps, check against numpy, table and chart. `--quick`, `--tile`, `--block`, `--threads`, `--affinity`, `--ghz` |
| `scripts/sweep_tiles.py` | Grid search over tile sizes (the article's tuning step) |
| `scripts/bench_numpy.py` | numpy/BLAS baseline, timed the same way as the kernels |
| `scripts/verify_numpy.py` | Checks a kernel's output against numpy |
| `scripts/plot.py` | Turns results.csv into summary.md and chart.png |
| `scripts/sysinfo.py` | Records CPU, caches, power mode and source, compilers, BLAS and git commit |
| `src/bench_gbench.cpp`, `Makefile` | Optional Google Benchmark version for Linux and macOS (`make gbench`) |

## Single runs

```powershell
.\build\mmm.exe reorder
.\build\mmm.exe tiled --tile 32
$env:OMP_NUM_THREADS = 8; .\build\mmm.exe parallel --tile 16 --block 256
.\build\mmm_O0.exe naive --min-reps 3
python scripts\build.py --vecreport
python scripts\build.py --asm
```

Every run checks its result against a float64 reference and prints `OK` or `WRONG RESULT`.
On Windows, the binaries opt out of power throttling (EcoQoS). Without that, Windows 11 can move them to efficiency cores when the terminal loses focus.
