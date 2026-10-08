// bench_gbench.cpp — OPTIONAL Google Benchmark version (what the article used).
// The main harness (bench.cpp) needs no dependencies; use this one if you want
// Google Benchmark's statistics/JSON output or to match the article exactly.
//
// Build:  make gbench
// Run:    ./build/mmm_gbench --benchmark_repetitions=5 \
//           --benchmark_report_aggregates_only=true \
//           --benchmark_out=results/gbench.json --benchmark_out_format=json
// Filter: ./build/mmm_gbench --benchmark_filter=LoopOrder
#include "kernels.h"

#include <benchmark/benchmark.h>
#include <cstring>
#include <random>
#include <vector>

#ifndef MMM_N
#define MMM_N 1024
#endif
constexpr int N = MMM_N;

namespace {

struct Mats {
  std::vector<float> A, B, C;
  Mats() : A((size_t)N * N), B((size_t)N * N), C((size_t)N * N) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    for (auto &x : A) x = d(rng);
    for (auto &x : B) x = d(rng);
  }
};

template <void (*Kernel)(const float *, const float *, float *)>
void BM(benchmark::State &state) {
  static Mats m;
  for (auto _ : state) {
    state.PauseTiming();  // zero the output outside the measured region
    std::memset(m.C.data(), 0, m.C.size() * sizeof(float));
    state.ResumeTiming();
    Kernel(m.A.data(), m.B.data(), m.C.data());
    benchmark::DoNotOptimize(m.C.data());
    benchmark::ClobberMemory();
  }
  // 2*N^3 FLOPs per iteration -> reported as FLOP/s
  state.counters["FLOPS"] = benchmark::Counter(
      2.0 * N * N * N, benchmark::Counter::kIsIterationInvariantRate,
      benchmark::Counter::kIs1000);
}

void naive(const float *a, const float *b, float *c) {
  matmulImplNaive<N, N, N>(a, b, c);
}
void regacc(const float *a, const float *b, float *c) {
  matmulImplNaiveRegisterAcc<N, N, N>(a, b, c);
}
void loopOrder(const float *a, const float *b, float *c) {
  matmulImplLoopOrder<N, N, N>(a, b, c);
}
void tiling16(const float *a, const float *b, float *c) {
  matmulImplTiling<N, N, N, 16>(a, b, c);
}
void parallel16(const float *a, const float *b, float *c) {
  matmulImplRowColParallelInnerTiling<N, N, N, 16, 256>(a, b, c);
}

}  // namespace

BENCHMARK(BM<naive>)->Name("Naive")->Unit(benchmark::kMillisecond)->Iterations(2);
BENCHMARK(BM<regacc>)->Name("NaiveRegisterAcc")->Unit(benchmark::kMillisecond)->Iterations(2);
BENCHMARK(BM<loopOrder>)->Name("LoopOrder")->Unit(benchmark::kMillisecond);
BENCHMARK(BM<tiling16>)->Name("LoopOrderTiling16")->Unit(benchmark::kMillisecond);
BENCHMARK(BM<parallel16>)->Name("RowColParallelTiling16")->Unit(benchmark::kMillisecond)->UseRealTime();

BENCHMARK_MAIN();
