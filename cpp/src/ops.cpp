#include "tfm/ops.hpp"

#include <cmath>
#include <cstddef>

#if defined(TFM_USE_EIGEN)
#include <Eigen/Dense>
#elif defined(__APPLE__)
// Use the modern Accelerate BLAS interface (macOS 13.3+ / iOS 16.4+). Without
// ACCELERATE_LAPACK_ILP64 this keeps 32-bit integer dimensions, so the
// cblas_sgemm signature is unchanged — only the deprecated classic entry
// points are avoided.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#endif

namespace tfm {

namespace {

/**
 * Row-major GEMM: C(M×N) = scale · A(M×K)·op(B) + bias, where
 *   op(B) = Bᵀ  if transB  (B stored N×K), else B (B stored K×N).
 *
 *   - Apple  → Accelerate cblas_sgemm (BLAS, multi-threaded)
 *   - Eigen  → Eigen Map product (SIMD, cross-platform; used on Android)
 *   - else   → portable scalar loop
 *
 * lda/ldb/ldc are the row strides, so head slices of a packed matrix (stride
 * = full d) can be GEMM'd in place without copying. bias may be nullptr.
 * BLAS/Eigen accumulation order differs from the scalar path by ~1e-6 — well
 * inside the 1e-3 parity tolerance.
 */
void gemm(const float* A, const float* B, const float* bias,
          int M, int K, int N, int lda, int ldb, int ldc, float* C, float scale,
          bool transB) {
#if defined(TFM_USE_EIGEN)
  using EM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using S = Eigen::Stride<Eigen::Dynamic, 1>;
  Eigen::Map<const EM, 0, S> Am(A, M, K, S(lda, 1));
  Eigen::Map<EM, 0, S> Cm(C, M, N, S(ldc, 1));
  if (transB) {
    Eigen::Map<const EM, 0, S> Bm(B, N, K, S(ldb, 1));
    Cm = (Am * Bm.transpose()) * scale;
  } else {
    Eigen::Map<const EM, 0, S> Bm(B, K, N, S(ldb, 1));
    Cm = (Am * Bm) * scale;
  }
  if (bias) {
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) C[(size_t)i * ldc + j] += bias[j];
  }
#elif defined(__APPLE__)
  if (bias) {
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) C[(size_t)i * ldc + j] = bias[j];
  } else {
    for (int i = 0; i < M; ++i)
      for (int j = 0; j < N; ++j) C[(size_t)i * ldc + j] = 0.0f;
  }
  if (M > 0 && N > 0 && K > 0) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, transB ? CblasTrans : CblasNoTrans,
                M, N, K, scale, A, lda, B, ldb, 1.0f, C, ldc);
  }
#else
  for (int i = 0; i < M; ++i) {
    const float* ai = A + (size_t)i * lda;
    for (int j = 0; j < N; ++j) {
      float acc = bias ? bias[j] : 0.0f;
      if (transB) {
        const float* bj = B + (size_t)j * ldb;
        for (int k = 0; k < K; ++k) acc += ai[k] * bj[k];
      } else {
        for (int k = 0; k < K; ++k) acc += ai[k] * B[(size_t)k * ldb + j];
      }
      C[(size_t)i * ldc + j] = acc * scale;
    }
  }
#endif
}

}  // namespace

void linear(const Mat& x, const Mat& W, const Vec& b, Mat& y) {
  const int M = x.R, K = x.C, N = W.R;
  y = Mat(M, N);
  gemm(x.d.data(), W.d.data(), b.data(), M, K, N, K, K, N, y.d.data(), 1.0f, true);
}

void linearSlice(const Mat& x, const Mat& W, const Vec& b,
                 int slice_off, int len, Mat& y) {
  const int M = x.R, K = x.C;
  y = Mat(M, len);
  gemm(x.d.data(), W.d.data() + (size_t)slice_off * K, b.data() + slice_off,
       M, K, len, K, K, len, y.d.data(), 1.0f, true);
}

void relu(Mat& x) {
  for (float& v : x.d) v = v > 0.0f ? v : 0.0f;
}

// Abramowitz & Stegun 7.1.26 — max abs error 1.5e-7, inlinable and
// branch-free, so the compiler auto-vectorizes it (std::erf is an opaque
// libm call). Well inside the 1e-3 parity tolerance vs torch's exact erf.
static inline float erfPoly(float x) {
  const float a1 = 0.254829592f, a2 = -0.284496736f, a3 = 1.421413741f;
  const float a4 = -1.453152027f, a5 = 1.061405429f, p = 0.3275911f;
  const float sign = x < 0.0f ? -1.0f : 1.0f;
  x = std::fabs(x);
  const float t = 1.0f / (1.0f + p * x);
  const float poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t;
  return sign * (1.0f - poly * std::exp(-x * x));
}

void geluExact(Mat& x) {
  const float inv_sqrt2 = 0.70710678118654752440f;  // 1/sqrt(2)
  for (float& v : x.d) {
    v = v * 0.5f * (1.0f + erfPoly(v * inv_sqrt2));
  }
}

void softmaxRows(Mat& x) {
#if defined(TFM_USE_EIGEN)
  using EM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<EM> X(x.d.data(), x.R, x.C);
  Eigen::Matrix<float, Eigen::Dynamic, 1> mx = X.rowwise().maxCoeff();
  for (int i = 0; i < x.R; ++i) X.row(i).array() -= mx(i);
  X = X.array().exp().matrix();
  Eigen::Matrix<float, Eigen::Dynamic, 1> rs = X.rowwise().sum();
  for (int i = 0; i < x.R; ++i) X.row(i).array() /= rs(i);
#elif defined(__APPLE__)
  for (int i = 0; i < x.R; ++i) {
    float* row = x.row(i);
    const int len = x.C;
    float m = row[0];
    for (int j = 1; j < len; ++j) m = std::max(m, row[j]);
    for (int j = 0; j < len; ++j) row[j] -= m;
    vvexpf(row, row, &len);  // SIMD exp (in-place)
    double sum = 0.0;
    for (int j = 0; j < len; ++j) sum += row[j];
    const float inv = (float)(1.0 / sum);
    for (int j = 0; j < len; ++j) row[j] *= inv;
  }
#else
  for (int i = 0; i < x.R; ++i) {
    float* row = x.row(i);
    float m = row[0];
    for (int j = 1; j < x.C; ++j) m = std::max(m, row[j]);
    double sum = 0.0;
    for (int j = 0; j < x.C; ++j) {
      row[j] = std::exp((double)row[j] - m);
      sum += row[j];
    }
    for (int j = 0; j < x.C; ++j) row[j] = (float)((double)row[j] / sum);
  }
#endif
}

void layerNorm(const Mat& x, const Vec& g, const Vec& b, float eps, Mat& y) {
  const int N = x.R, C = x.C;
  y = Mat(N, C);
  for (int i = 0; i < N; ++i) {
    const float* xi = x.row(i);
    double mean = 0.0;
    for (int j = 0; j < C; ++j) mean += xi[j];
    mean /= C;
    double var = 0.0;
    for (int j = 0; j < C; ++j) {
      const double d = (double)xi[j] - mean;
      var += d * d;
    }
    var /= C;
    const double inv = 1.0 / std::sqrt(var + eps);
    float* yi = y.row(i);
    for (int j = 0; j < C; ++j) {
      yi[j] = (float)(((double)xi[j] - mean) * inv) * g[j] + b[j];
    }
  }
}

void meanRows(const Mat& x, Mat& out) {
  out = Mat(1, x.C);
  for (int j = 0; j < x.C; ++j) {
    double acc = 0.0;
    for (int i = 0; i < x.R; ++i) acc += x.at(i, j);
    out.at(0, j) = (float)(acc / x.R);
  }
}

void attention(const Mat& x, int nhead, float scale,
               const Mat& inProjW, const Vec& inProjB,
               const Mat& outProjW, const Vec& outProjB, Mat& out) {
  const int N = x.R, d = x.C, hd = d / nhead;

  Mat q, k, v;
  linearSlice(x, inProjW, inProjB, 0, d, q);
  linearSlice(x, inProjW, inProjB, d, d, k);
  linearSlice(x, inProjW, inProjB, 2 * d, d, v);

  // Heads are independent; each gets its own `scores` so the loop is safe to
  // run in parallel (OpenMP on Android). Each head writes a disjoint slice of
  // `o`, so no locks needed.
  Mat o(N, d, 0.0f);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
  for (int h = 0; h < nhead; ++h) {
    const int base = h * hd;
    Mat scores(N, N);
    // scores = scale · Q_h · K_hᵀ   (head slice: lda/ldb = full d)
    gemm(q.d.data() + base, k.d.data() + base, nullptr, N, hd, N, d, d, N,
         scores.d.data(), scale, true);
    softmaxRows(scores);
    // o_h = P_h · V_h   (V_h stored K×N: K=N tokens, N=hd, stride d)
    gemm(scores.d.data(), v.d.data() + base, nullptr, N, N, hd, N, d, d,
         o.d.data() + base, 1.0f, false);
  }
  linear(o, outProjW, outProjB, out);
}

}  // namespace tfm