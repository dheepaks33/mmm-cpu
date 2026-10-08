# MMM-on-CPU results

- CPU: Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz
- Compiler: Clang 22.1.8 (https://github.com/msys2/MINGW-packages d34feaaa1d0aff624c870910b6f2589aa40d20de)
- Flags used: -O0  |  -O3 -march=native -ffast-math -fopenmp
- N = 1024 (2*N^3 = 2.15 GFLOP per multiply)
- CPU affinity mask: 0xFF
- Timestamp (UTC): 2026-10-08T05:56:14Z

| Step | Kernel | Threads | Median ms | Stdev ms | GFLOP/s | FLOPs/core/cycle | vs prev | vs first row | % of numpy | Article ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 naive (-O0) | naive | 1 | 4431.7 | 36.0 | 0.5 | 0.20 | - | 1x | 0.2% | 4481 |
| 2 naive + flags | naive | 1 | 1921.5 | 4.7 | 1.1 | 0.47 | 2.3x | 2x | 0.4% | 1621 |
| 3 + register acc | regacc | 1 | 1929.0 | 4.0 | 1.1 | 0.46 | 1.0x | 2x | 0.4% | 1512 |
| 4 loop reorder (RIC) | reorder | 1 | 86.3 | 28.0 | 24.9 | 10.37 | 22.4x | 51x | 9.2% | 89 |
| 5 + L1 tiling (16) | tiled t=16 | 1 | 62.0 | 3.5 | 34.7 | 14.44 | 1.4x | 72x | 13% | 70 |
| 6 + multithreading | parallel t=16 | 4 | 27.0 | 5.2 | 79.5 | 8.28 | 2.3x | 164x | 30% | 16 |
| 7 numpy (BLAS) | numpy | 4 | 8.0 | 1.8 | 269.2 | 28.04 | 3.4x | 555x | 100% | 8 |
