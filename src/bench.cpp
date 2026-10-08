// bench.cpp — dependency-free benchmark harness for the kernels in kernels.h.
// Builds on Windows (clang/MinGW), Linux and macOS.
//
// Usage:
//   mmm <kernel> [options]          (build\mmm.exe on Windows, ./build/mmm elsewhere)
//   mmm topology                    show core types and a P-core affinity mask
//
// Kernels:  naive | regacc | reorder | tiled | parallel
// Options:
//   --tile T        inner tile size for tiled/parallel (1,2,4,8,16,32,64,128,256)
//   --block B       row/col block size for parallel (64,128,256)  [default 256]
//   --min-time S    keep repeating until this many seconds measured [default 2]
//   --min-reps R    at least this many timed repetitions           [default 5]
//   --max-reps R    stop after this many repetitions               [default 100]
//   --csv FILE      append one result row to FILE (header written if new)
//   --label TEXT    free-text label stored in the CSV (e.g. "4 loop reorder")
//   --dump DIR      write A.bin, B.bin, C.bin (raw float32) for verify_numpy.py
//
// Method: fixed-seed random inputs in [-1, 1]; one warm-up run (skipped if a
// single run takes > 1 s); `result` zeroed OUTSIDE the timer before every
// repetition; wall-clock std::chrono::steady_clock around the kernel call
// only. Reports min / median / mean / stdev; GFLOP/s uses the median and
// 2*N^3 FLOPs. Every run is checked against a float64 reference.
//
// Windows only: the process opts out of power throttling (EcoQoS) at start-up,
// so Windows 11 does not move it to efficiency cores when its window loses focus.
#if defined(_WIN32) && !defined(_CRT_SECURE_NO_WARNINGS)
#define _CRT_SECURE_NO_WARNINGS
#endif
#include "kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32)
#  ifndef _WIN32_WINNT
#    define _WIN32_WINNT 0x0A00
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <malloc.h>
#elif defined(__APPLE__)
#  include <sys/sysctl.h>
#  include <unistd.h>
#else
#  include <sched.h>
#  include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#  define MMM_X86 1
#  if defined(_MSC_VER) && !defined(__clang__)
#    include <intrin.h>
#  else
#    include <cpuid.h>
#  endif
#endif

// Matrix size. Passed as -DMMM_N=... (a bare N would collide with windows.h).
#ifndef MMM_N
#define MMM_N 1024
#endif
#ifndef BUILD_FLAGS
#define BUILD_FLAGS "unknown"
#endif

#if defined(__VERSION__)
#define COMPILER_VERSION __VERSION__
#elif defined(_MSC_FULL_VER)
#define MMM_STR2(x) #x
#define MMM_STR(x) MMM_STR2(x)
#define COMPILER_VERSION "MSVC " MMM_STR(_MSC_FULL_VER)
#else
#define COMPILER_VERSION "unknown"
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif

namespace fs = std::filesystem;
constexpr int N = MMM_N;
constexpr int ROWS = N, COLS = N, INNERS = N;
using KernelFn = void (*)(const float *, const float *, float *);

// noinline wrappers: keep each kernel a separate, un-hoistable call so the
// compiler cannot merge repetitions or optimise across the timing boundary.
NOINLINE void k_naive(const float *a, const float *b, float *c) {
  matmulImplNaive<ROWS, COLS, INNERS>(a, b, c);
}
NOINLINE void k_regacc(const float *a, const float *b, float *c) {
  matmulImplNaiveRegisterAcc<ROWS, COLS, INNERS>(a, b, c);
}
NOINLINE void k_reorder(const float *a, const float *b, float *c) {
  matmulImplLoopOrder<ROWS, COLS, INNERS>(a, b, c);
}
template <int T>
NOINLINE void k_tiled(const float *a, const float *b, float *c) {
  matmulImplTiling<ROWS, COLS, INNERS, T>(a, b, c);
}
template <int T, int B>
NOINLINE void k_parallel(const float *a, const float *b, float *c) {
  matmulImplRowColParallelInnerTiling<ROWS, COLS, INNERS, T, B>(a, b, c);
}

// Tile sizes must be compile-time constants (template params), so dispatch
// over a fixed menu of them.
static KernelFn pick_tiled(int t) {
  switch (t) {
  case 1: return k_tiled<1>;     case 2: return k_tiled<2>;
  case 4: return k_tiled<4>;     case 8: return k_tiled<8>;
  case 16: return k_tiled<16>;   case 32: return k_tiled<32>;
  case 64: return k_tiled<64>;   case 128: return k_tiled<128>;
  case 256: return k_tiled<256>;
  }
  return nullptr;
}
template <int B> static KernelFn pick_parallel_b(int t) {
  switch (t) {
  case 1: return k_parallel<1, B>;     case 2: return k_parallel<2, B>;
  case 4: return k_parallel<4, B>;     case 8: return k_parallel<8, B>;
  case 16: return k_parallel<16, B>;   case 32: return k_parallel<32, B>;
  case 64: return k_parallel<64, B>;   case 128: return k_parallel<128, B>;
  case 256: return k_parallel<256, B>;
  }
  return nullptr;
}
static KernelFn pick_parallel(int t, int b) {
  if (N % b != 0) return nullptr;
  switch (b) {
  case 64: return pick_parallel_b<(N % 64 == 0) ? 64 : N>(t);
  case 128: return pick_parallel_b<(N % 128 == 0) ? 128 : N>(t);
  case 256: return pick_parallel_b<(N % 256 == 0) ? 256 : N>(t);
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// Platform helpers
// ---------------------------------------------------------------------------
static double now_s() {
  using clk = std::chrono::steady_clock;
  return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

static std::string trim(std::string s) {
  while (!s.empty() && (s.back() == '\n' || s.back() == ' ' || s.back() == '\r')) s.pop_back();
  size_t i = 0;
  while (i < s.size() && s[i] == ' ') i++;
  return s.substr(i);
}

#ifdef MMM_X86
static void cpuid_regs(unsigned leaf, unsigned sub, unsigned r[4]) {
#if defined(_MSC_VER) && !defined(__clang__)
  int t[4];
  __cpuidex(t, (int)leaf, (int)sub);
  for (int i = 0; i < 4; i++) r[i] = (unsigned)t[i];
#else
  __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}

static std::string cpuid_brand() {
  unsigned regs[12] = {0}, r[4];
  cpuid_regs(0x80000000u, 0, r);
  if (r[0] < 0x80000004u) return "";
  for (unsigned i = 0; i < 3; i++) cpuid_regs(0x80000002u + i, 0, regs + 4 * i);
  char buf[49];
  memcpy(buf, regs, 48);
  buf[48] = 0;
  return trim(buf);
}

static std::string simd_features() {
  unsigned r1[4], r7[4] = {0, 0, 0, 0}, r0[4];
  cpuid_regs(0, 0, r0);
  cpuid_regs(1, 0, r1);
  if (r0[0] >= 7) cpuid_regs(7, 0, r7);
  auto yn = [](bool b) { return b ? "yes" : "no"; };
  char buf[160];
  snprintf(buf, sizeof(buf), "AVX %s, AVX2 %s, FMA %s, AVX-512F %s",
           yn(r1[2] & (1u << 28)), yn(r7[1] & (1u << 5)), yn(r1[2] & (1u << 12)),
           yn(r7[1] & (1u << 16)));
  return buf;
}
#else
static std::string simd_features() {
#if defined(__aarch64__) || defined(_M_ARM64)
  return "NEON (ARM64)";
#else
  return "unknown";
#endif
}
#endif

static std::string cpu_model() {
#if defined(__APPLE__)
  char buf[256];
  size_t len = sizeof(buf);
  if (sysctlbyname("machdep.cpu.brand_string", buf, &len, nullptr, 0) == 0)
    return std::string(buf);
  return "unknown";
#elif defined(_WIN32)
#ifdef MMM_X86
  std::string b = cpuid_brand();
  if (!b.empty()) return b;
#endif
  const char *id = getenv("PROCESSOR_IDENTIFIER");
  return id ? id : "unknown";
#else
  FILE *f = fopen("/proc/cpuinfo", "r");
  if (!f) return "unknown";
  char line[512];
  std::string model = "unknown";
  while (fgets(line, sizeof(line), f)) {
    if (strncmp(line, "model name", 10) == 0) {
      char *p = strchr(line, ':');
      if (p) model = trim(std::string(p + 1));
      break;
    }
  }
  fclose(f);
#ifdef MMM_X86
  if (model == "unknown") model = cpuid_brand();
#endif
  return model;
#endif
}

static std::string host_name() {
#if defined(_WIN32)
  const char *h = getenv("COMPUTERNAME");
  return h ? h : "unknown";
#else
  char host[256] = "unknown";
  gethostname(host, sizeof(host));
  host[sizeof(host) - 1] = 0;
  return host;
#endif
}

// Which logical CPUs this process may run on, as a hex mask of the first 64.
static std::string affinity_mask() {
  char buf[32];
#if defined(_WIN32)
  DWORD_PTR proc = 0, sys = 0;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys)) return "unknown";
  snprintf(buf, sizeof(buf), "0x%llX", (unsigned long long)proc);
  return buf;
#elif defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) != 0) return "unknown";
  unsigned long long m = 0;
  for (int i = 0; i < 64; i++)
    if (CPU_ISSET(i, &set)) m |= 1ull << i;
  snprintf(buf, sizeof(buf), "0x%llX", m);
  return buf;
#else
  return "n/a";
#endif
}

// Windows 11 runs background / unfocused processes with "EcoQoS", which on
// hybrid Intel CPUs means efficiency cores and lower clocks. Opt out.
static void disable_power_throttling() {
#if defined(_WIN32)
  PROCESS_POWER_THROTTLING_STATE s;
  memset(&s, 0, sizeof(s));
  s.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
  s.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
  s.StateMask = 0;  // 0 = never throttle execution speed
  using SetPI = BOOL(WINAPI *)(HANDLE, PROCESS_INFORMATION_CLASS, LPVOID, DWORD);
  // Looked up at run time so the binary still starts on very old Windows.
  auto fn = reinterpret_cast<SetPI>(reinterpret_cast<void *>(
      GetProcAddress(GetModuleHandleA("kernel32.dll"), "SetProcessInformation")));
  if (fn) fn(GetCurrentProcess(), ProcessPowerThrottling, &s, sizeof(s));
#endif
}

// `mmm topology`: list core types so you can pin step 6 to performance cores.
static int print_topology() {
  printf("CPU: %s\n", cpu_model().c_str());
  printf("SIMD: %s\n", simd_features().c_str());
  printf("Logical processors: %u   current affinity: %s\n",
         std::thread::hardware_concurrency(), affinity_mask().c_str());
#if defined(_WIN32)
  // Looked up at run time: some MinGW headers lack the declaration.
  using GetCpuSets = BOOL(WINAPI *)(PSYSTEM_CPU_SET_INFORMATION, ULONG, PULONG, HANDLE, ULONG);
  auto getCpuSets = reinterpret_cast<GetCpuSets>(reinterpret_cast<void *>(
      GetProcAddress(GetModuleHandleA("kernel32.dll"), "GetSystemCpuSetInformation")));
  ULONG len = 0;
  if (getCpuSets) getCpuSets(nullptr, 0, &len, GetCurrentProcess(), 0);
  if (!getCpuSets || len == 0) {
    printf("GetSystemCpuSetInformation not available on this Windows build.\n");
    return 0;
  }
  std::vector<unsigned char> buf(len);
  auto *first = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data());
  if (!getCpuSets(first, len, &len, GetCurrentProcess(), 0)) {
    printf("GetSystemCpuSetInformation failed (error %lu)\n", (unsigned long)GetLastError());
    return 1;
  }
  // core index -> (efficiency class, list of logical processors)
  std::map<int, std::pair<int, std::vector<int>>> cores;
  int maxClass = 0, minClass = 255;
  bool otherGroups = false;
  for (ULONG off = 0; off < len;) {
    auto *e = reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(buf.data() + off);
    if (e->Size == 0) break;
    if (e->Type == CpuSetInformation) {
      if (e->CpuSet.Group != 0) {
        otherGroups = true;
      } else {
        auto &c = cores[e->CpuSet.CoreIndex];
        c.first = e->CpuSet.EfficiencyClass;
        c.second.push_back(e->CpuSet.LogicalProcessorIndex);
        maxClass = std::max<int>(maxClass, e->CpuSet.EfficiencyClass);
        minClass = std::min<int>(minClass, e->CpuSet.EfficiencyClass);
      }
    }
    off += e->Size;
  }
  printf("\n  core  class  logical processors\n");
  for (auto &kv : cores) {
    printf("  %4d  %5d  ", kv.first, kv.second.first);
    for (int lp : kv.second.second) printf("%d ", lp);
    printf("%s\n", kv.second.first == maxClass && maxClass != minClass ? " <- P-core" : "");
  }
  if (otherGroups) printf("  (processors outside group 0 not shown)\n");
  if (cores.empty()) return 0;
  if (maxClass == minClass) {
    printf("\nAll %zu cores are the same type: no pinning needed.\n", cores.size());
    printf("Suggested: --threads %zu (one per physical core)", cores.size());
    if (std::thread::hardware_concurrency() != cores.size())
      printf("; also try --threads %u (hyperthreads)", std::thread::hardware_concurrency());
    printf(".\n");
    return 0;
  }
  unsigned long long mask = 0;
  int pcores = 0, pthreads = 0;
  for (auto &kv : cores)
    if (kv.second.first == maxClass) {
      pcores++;
      for (int lp : kv.second.second) {
        if (lp < 64) mask |= 1ull << lp;
        pthreads++;
      }
    }
  printf("\nHybrid CPU: %d performance cores (%d threads), %zu other cores.\n",
         pcores, pthreads, cores.size() - pcores);
  printf("Pin the benchmark to the P-cores:\n");
  printf("  python scripts\\run_all.py --affinity 0x%llX --threads %d\n", mask, pcores);
  printf("Also try --threads %d (both hyperthreads of each P-core).\n", pthreads);
#elif defined(__linux__)
  FILE *p = fopen("/sys/devices/cpu_core/cpus", "r");
  FILE *e = fopen("/sys/devices/cpu_atom/cpus", "r");
  if (p && e) {
    char pb[256] = {0}, eb[256] = {0};
    if (!fgets(pb, sizeof(pb), p)) pb[0] = 0;
    if (!fgets(eb, sizeof(eb), e)) eb[0] = 0;
    printf("Hybrid CPU. P-core CPUs: %s   E-core CPUs: %s\n", trim(pb).c_str(), trim(eb).c_str());
    printf("Pin with: taskset -c %s python3 scripts/run_all.py\n", trim(pb).c_str());
  } else {
    printf("No hybrid topology exposed (all cores same type, or a VM/WSL2). See `lscpu --all --extended`.\n");
  }
  if (p) fclose(p);
  if (e) fclose(e);
#else
  printf("On macOS: sysctl hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu\n");
#endif
  return 0;
}

static std::string csv_escape(std::string s) {
  for (auto &ch : s)
    if (ch == '"') ch = '\'';
  return "\"" + s + "\"";
}

static float *alloc_matrix(size_t n) {
  // 64-byte alignment = one cache line; size rounded up to a multiple of 64.
  size_t bytes = ((n * sizeof(float) + 63) / 64) * 64;
#if defined(_WIN32)
  float *p = static_cast<float *>(_aligned_malloc(bytes, 64));
#else
  float *p = static_cast<float *>(aligned_alloc(64, bytes));
#endif
  if (!p) { fprintf(stderr, "allocation failed\n"); exit(1); }
  return p;
}

static void free_matrix(float *p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  free(p);
#endif
}

static void write_bin(const fs::path &path, const float *p, size_t n) {
  FILE *f = fopen(path.string().c_str(), "wb");
  if (!f) { perror(path.string().c_str()); exit(1); }
  fwrite(p, sizeof(float), n, f);
  fclose(f);
}

static void usage() {
  fprintf(stderr,
          "usage: mmm <naive|regacc|reorder|tiled|parallel> [--tile T] "
          "[--block B] [--min-time S] [--min-reps R] [--max-reps R] "
          "[--csv FILE] [--label TEXT] [--dump DIR]\n"
          "       mmm topology\n");
  exit(2);
}

int main(int argc, char **argv) {
  if (argc < 2) usage();
  std::string kernel = argv[1];
  if (kernel == "topology" || kernel == "--topology") return print_topology();
  disable_power_throttling();

  int tile = 16, block = 256, min_reps = 5, max_reps = 100;
  double min_time = 2.0;
  std::string csv_path, label, dump_dir;
  for (int i = 2; i < argc; i++) {
    std::string a = argv[i];
    auto next = [&]() -> const char * {
      if (i + 1 >= argc) usage();
      return argv[++i];
    };
    if (a == "--tile") tile = atoi(next());
    else if (a == "--block") block = atoi(next());
    else if (a == "--min-time") min_time = atof(next());
    else if (a == "--min-reps") min_reps = atoi(next());
    else if (a == "--max-reps") max_reps = atoi(next());
    else if (a == "--csv") csv_path = next();
    else if (a == "--label") label = next();
    else if (a == "--dump") dump_dir = next();
    else usage();
  }

  KernelFn fn = nullptr;
  bool uses_tile = false, is_parallel = false;
  if (kernel == "naive") fn = k_naive;
  else if (kernel == "regacc") fn = k_regacc;
  else if (kernel == "reorder") fn = k_reorder;
  else if (kernel == "tiled") { fn = pick_tiled(tile); uses_tile = true; }
  else if (kernel == "parallel") {
    fn = pick_parallel(tile, block);
    uses_tile = true;
    is_parallel = true;
  } else usage();
  if (!fn) {
    fprintf(stderr, "unsupported tile/block size (tile=%d block=%d, N=%d)\n",
            tile, block, N);
    return 2;
  }

  int threads = 1;
#ifdef _OPENMP
  if (is_parallel) threads = omp_get_max_threads();
#else
  if (is_parallel)
    fprintf(stderr, "warning: built without OpenMP, running single-threaded\n");
#endif

  const size_t nA = (size_t)ROWS * INNERS, nB = (size_t)INNERS * COLS,
               nC = (size_t)ROWS * COLS;
  float *A = alloc_matrix(nA), *B = alloc_matrix(nB), *C = alloc_matrix(nC);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (size_t i = 0; i < nA; i++) A[i] = dist(rng);
  for (size_t i = 0; i < nB; i++) B[i] = dist(rng);

  // float64 reference, cache-friendly loop order so it stays quick.
  std::vector<double> ref(nC, 0.0);
  for (int r = 0; r < ROWS; r++)
    for (int k = 0; k < INNERS; k++) {
      double a = A[(size_t)r * INNERS + k];
      const float *brow = B + (size_t)k * COLS;
      double *crow = ref.data() + (size_t)r * COLS;
      for (int c = 0; c < COLS; c++) crow[c] += a * brow[c];
    }

  // Warm-up (page faults, caches, thread pool start-up, CPU clock ramp).
  std::vector<double> times;
  memset(C, 0, nC * sizeof(float));
  double t0 = now_s();
  fn(A, B, C);
  double first = now_s() - t0;
  if (first > 1.0) times.push_back(first);  // slow kernel: count it as rep 1

  double measured = 0.0;
  for (double t : times) measured += t;
  int effective_min_reps = first > 1.0 ? std::min(min_reps, 3) : min_reps;
  while ((int)times.size() < max_reps &&
         ((int)times.size() < effective_min_reps || measured < min_time)) {
    memset(C, 0, nC * sizeof(float));
    double s = now_s();
    fn(A, B, C);
    double dt = now_s() - s;
    times.push_back(dt);
    measured += dt;
  }

  // Correctness against the float64 reference (C holds the last rep).
  double max_abs = 0.0, max_ref = 0.0;
  for (size_t i = 0; i < nC; i++) {
    max_abs = std::max(max_abs, std::fabs((double)C[i] - ref[i]));
    max_ref = std::max(max_ref, std::fabs(ref[i]));
  }
  double rel = max_abs / (max_ref > 0 ? max_ref : 1.0);
  bool ok = rel < 1e-4;

  std::vector<double> sorted = times;
  std::sort(sorted.begin(), sorted.end());
  size_t n = sorted.size();
  double median = n % 2 ? sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);
  double mean = 0;
  for (double t : times) mean += t;
  mean /= n;
  double var = 0;
  for (double t : times) var += (t - mean) * (t - mean);
  double stdev = n > 1 ? std::sqrt(var / (n - 1)) : 0.0;
  double flops = 2.0 * ROWS * COLS * INNERS;
  double gflops = flops / median / 1e9;

  std::string kname = kernel;
  if (uses_tile) kname += " tile=" + std::to_string(tile);
  if (is_parallel) kname += " block=" + std::to_string(block);
  printf("%-34s N=%d threads=%-2d reps=%-3d  median %9.2f ms  min %9.2f ms  "
         "stdev %6.2f ms  %7.2f GFLOP/s  rel_err %.1e %s\n",
         kname.c_str(), N, threads, (int)n, median * 1e3, sorted[0] * 1e3,
         stdev * 1e3, gflops, rel, ok ? "OK" : "WRONG RESULT");
  fflush(stdout);

  if (!dump_dir.empty()) {
    fs::path d(dump_dir);
    fs::create_directories(d);
    write_bin(d / "A.bin", A, nA);
    write_bin(d / "B.bin", B, nB);
    write_bin(d / "C.bin", C, nC);
    FILE *f = fopen((d / "shape.txt").string().c_str(), "w");
    if (f) { fprintf(f, "%d %d %d\n", ROWS, COLS, INNERS); fclose(f); }
  }

  if (!csv_path.empty()) {
    bool fresh = !fs::exists(csv_path);
    FILE *f = fopen(csv_path.c_str(), "ab");  // binary: LF endings on every OS
    if (!f) { perror(csv_path.c_str()); return 1; }
    if (fresh)
      fprintf(f, "timestamp_utc,label,host,cpu,compiler,flags,n,kernel,tile,"
                 "block,threads,reps,min_ms,median_ms,mean_ms,stdev_ms,"
                 "gflops,rel_err,status,affinity\n");
    char ts[32];
    time_t now = time(nullptr);
    strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", gmtime(&now));
    fprintf(f, "%s,%s,%s,%s,%s,%s,%d,%s,%s,%s,%d,%d,%.4f,%.4f,%.4f,%.4f,%.3f,%.2e,%s,%s\n",
            ts, csv_escape(label).c_str(), csv_escape(host_name()).c_str(),
            csv_escape(cpu_model()).c_str(), csv_escape(COMPILER_VERSION).c_str(),
            csv_escape(BUILD_FLAGS).c_str(), N, kernel.c_str(),
            uses_tile ? std::to_string(tile).c_str() : "",
            is_parallel ? std::to_string(block).c_str() : "", threads, (int)n,
            sorted[0] * 1e3, median * 1e3, mean * 1e3, stdev * 1e3, gflops,
            rel, ok ? "ok" : "WRONG", affinity_mask().c_str());
    fclose(f);
  }

  free_matrix(A); free_matrix(B); free_matrix(C);
  return ok ? 0 : 1;
}
