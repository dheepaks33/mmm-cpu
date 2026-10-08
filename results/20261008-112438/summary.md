# MMM-on-CPU results

- CPU: Intel(R) Core(TM) i5-9300H CPU @ 2.40GHz
- Compiler: Clang 22.1.8 (https://github.com/msys2/MINGW-packages d34feaaa1d0aff624c870910b6f2589aa40d20de)
- Flags used: -O0  |  -O3 -march=native -ffast-math -fopenmp
- N = 1024 (2*N^3 = 2.15 GFLOP per multiply)
- CPU affinity mask: 0xFF
- Timestamp (UTC): 2026-10-08T05:55:08Z

| Step | Kernel | Threads | Median ms | Stdev ms | GFLOP/s | FLOPs/core/cycle | vs prev | vs first row | % of numpy | Article ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 naive (-O0) | naive | 1 | 5315.4 | 496.1 | 0.4 | 0.02 | - | 1x | 0.2% | 4481 |
| 2 naive + flags | naive | 1 | 1938.9 | 34.7 | 1.1 | 0.05 | 2.7x | 3x | 0.4% | 1621 |
| 3 + register acc | regacc | 1 | 1942.1 | 16.9 | 1.1 | 0.05 | 1.0x | 3x | 0.4% | 1512 |
| 4 loop reorder (RIC) | reorder | 1 | 78.8 | 5.0 | 27.3 | 1.14 | 24.7x | 67x | 10% | 89 |
| 5 + L1 tiling (16) | tiled t=16 | 1 | 63.2 | 4.1 | 34.0 | 1.41 | 1.2x | 84x | 13% | 70 |
| 6 + multithreading | parallel t=16 | 8 | 19.5 | 3.9 | 110.1 | 1.15 | 3.2x | 273x | 42% | 16 |
| 7 numpy (BLAS) | numpy | 8 | 8.3 | 1.9 | 260.3 | 2.71 | 2.4x | 644x | 100% | 8 |
