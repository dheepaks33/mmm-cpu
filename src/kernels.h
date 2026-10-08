// kernels.h — the six matrix-multiply kernels from Simon Boehm's
// "Fast Multidimensional Matrix Multiplication on CPU from Scratch" (2022).
// https://siboehm.com/articles/22/Fast-MMM-on-CPU
//
// Layout: all matrices are row-major float32.
//   left   : rows   x inners
//   right  : inners x columns
//   result : rows   x columns
// Kernels that use `+=` expect `result` to be zeroed before the call
// (the benchmark harness does this outside the timed region).
//
// One deliberate fix vs. the article: the article indexes `left` with
// `row * columns + inner` in the first kernels. That is only correct for
// square matrices, so here it is `row * inners + inner` everywhere. For the
// 1024x1024 case the generated code is identical.
#pragma once
#include <algorithm>

// ---------------------------------------------------------------------------
// Step 1/2: Naive, loop order Row-Column-Inner (RCI).
// Inner loop walks DOWN a column of `right` -> a new cache line every step.
// ---------------------------------------------------------------------------
template <int rows, int columns, int inners>
inline void matmulImplNaive(const float *left, const float *right,
                            float *result) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      for (int inner = 0; inner < inners; inner++) {
        result[row * columns + col] +=
            left[row * inners + inner] * right[inner * columns + col];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Step 3: Naive + accumulate the dot product in a register, store once.
// ---------------------------------------------------------------------------
template <int rows, int columns, int inners>
inline void matmulImplNaiveRegisterAcc(const float *left, const float *right,
                                       float *result) {
  for (int row = 0; row < rows; row++) {
    for (int col = 0; col < columns; col++) {
      float acc = 0.0f;
      for (int inner = 0; inner < inners; inner++) {
        acc += left[row * inners + inner] * right[inner * columns + col];
      }
      result[row * columns + col] = acc;
    }
  }
}

// ---------------------------------------------------------------------------
// Step 4: Cache-aware loop order Row-Inner-Column (RIC).
// Inner loop now streams contiguously through a row of `right` and `result`,
// which also lets the compiler vectorize it (broadcast A, FMA 8/16 lanes).
// ---------------------------------------------------------------------------
template <int rows, int columns, int inners>
inline void matmulImplLoopOrder(const float *left, const float *right,
                                float *result) {
  for (int row = 0; row < rows; row++) {
    for (int inner = 0; inner < inners; inner++) {
      for (int col = 0; col < columns; col++) {
        result[row * columns + col] +=
            left[row * inners + inner] * right[inner * columns + col];
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Step 5: RIC + tiling (cache blocking) on the inner dimension.
// Only `tileSize` rows of `right` are live per pass, so they stay in L1/L2
// while every row of the output reuses them.
// ---------------------------------------------------------------------------
template <int rows, int columns, int inners, int tileSize>
inline void matmulImplTiling(const float *left, const float *right,
                             float *result) {
  for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
    for (int row = 0; row < rows; row++) {
      int innerTileEnd = std::min(inners, innerTile + tileSize);
      for (int inner = innerTile; inner < innerTileEnd; inner++) {
        for (int column = 0; column < columns; column++) {
          result[row * columns + column] +=
              left[row * inners + inner] * right[inner * columns + column];
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Step 6: RIC + inner tiling + OpenMP over (rowTile, columnTile) blocks.
// Each thread owns a disjoint block of `result`, so no atomics/locks needed.
// The article hard-codes num_threads(8); here the thread count comes from
// OMP_NUM_THREADS so it fits whatever laptop you run on.
// blockSize = 256 matches the article (1024/256 = 4x4 = 16 work items).
// ---------------------------------------------------------------------------
template <int rows, int columns, int inners, int tileSize, int blockSize>
inline void matmulImplRowColParallelInnerTiling(const float *left,
                                                const float *right,
                                                float *result) {
  static_assert(rows % blockSize == 0 && columns % blockSize == 0,
                "rows and columns must be multiples of blockSize");
#pragma omp parallel for shared(result, left, right) default(none) collapse(2)
  for (int rowTile = 0; rowTile < rows; rowTile += blockSize) {
    for (int columnTile = 0; columnTile < columns; columnTile += blockSize) {
      for (int innerTile = 0; innerTile < inners; innerTile += tileSize) {
        for (int row = rowTile; row < rowTile + blockSize; row++) {
          int innerTileEnd = std::min(inners, innerTile + tileSize);
          for (int inner = innerTile; inner < innerTileEnd; inner++) {
            for (int col = columnTile; col < columnTile + blockSize; col++) {
              result[row * columns + col] +=
                  left[row * inners + inner] * right[inner * columns + col];
            }
          }
        }
      }
    }
  }
}
