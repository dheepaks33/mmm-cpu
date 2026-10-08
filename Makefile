# Makefile — Linux / macOS / WSL2 alternative to `python scripts/build.py`
# (same flags). On Windows use scripts/build.py instead.
#
#   make                 -> build/mmm_O0 (step 1) and build/mmm (steps 2-6)
#   make gbench          -> build/mmm_gbench (optional, Google Benchmark, like the article)
#   make asm             -> build/bench.s (inspect the generated assembly)
#   make vecreport       -> print which loops the compiler vectorised
#   make N=2048          -> different matrix size (must be a multiple of 256)
#   make CXX=g++         -> use GCC instead of clang (article used clang 14)

CXX      := clang++
N        ?= 1024
BASE     := -std=c++17 -Wall -DMMM_N=$(N)
OPT      := -O3 -march=native -ffast-math
OMP      := -fopenmp

UNAME_S  := $(shell uname -s)
UNAME_M  := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
  # Apple clang has no bundled OpenMP: `brew install libomp` first.
  BREW_OMP := $(shell brew --prefix libomp 2>/dev/null)
  OMP      := -Xpreprocessor -fopenmp -I$(BREW_OMP)/include -L$(BREW_OMP)/lib -lomp
  ifeq ($(UNAME_M),arm64)
    # -march=native is not accepted by every Apple clang on arm64.
    # apple-m1 runs on all M-series chips; try -mcpu=native if your clang accepts it.
    OPT := -O3 -mcpu=apple-m1 -ffast-math
  endif
endif

.PHONY: all clean gbench asm vecreport

all: build/mmm_O0 build/mmm

build:
	mkdir -p build

# Step 1: "default flags" = no optimisation, exactly like the article's first row.
build/mmm_O0: src/bench.cpp src/kernels.h | build
	$(CXX) $(BASE) -O0 -Wno-unknown-pragmas -DBUILD_FLAGS='"-O0"' src/bench.cpp -o $@

# Steps 2-6: -O3 -march=native -ffast-math (+ OpenMP for step 6).
build/mmm: src/bench.cpp src/kernels.h | build
	$(CXX) $(BASE) $(OPT) $(OMP) -DBUILD_FLAGS='"$(OPT) -fopenmp"' src/bench.cpp -o $@

# Optional: the article benchmarked with Google Benchmark.
# Linux: sudo apt install libbenchmark-dev   macOS: brew install google-benchmark
gbench: build/mmm_gbench
build/mmm_gbench: src/bench_gbench.cpp src/kernels.h | build
	$(CXX) $(BASE) $(OPT) $(OMP) src/bench_gbench.cpp -o $@ -lbenchmark -lpthread $(GBENCH_FLAGS)

asm: | build
	$(CXX) $(BASE) $(OPT) $(OMP) -S -fno-asynchronous-unwind-tables src/bench.cpp -o build/bench.s
	@echo "wrote build/bench.s — search for k_reorder / k_naive and look for vfmadd / fmla"

vecreport: | build
ifeq ($(findstring clang,$(CXX)),clang)
	$(CXX) $(BASE) $(OPT) $(OMP) -c src/bench.cpp -o /dev/null -Rpass=loop-vectorize -Rpass-missed=loop-vectorize 2>&1 | grep "kernels.h.*remark" | sort | uniq
else
	$(CXX) $(BASE) $(OPT) $(OMP) -c src/bench.cpp -o /dev/null -fopt-info-vec-optimized -fopt-info-vec-missed 2>&1 | grep kernels.h | sort | uniq | head -40
endif

clean:
	rm -rf build
