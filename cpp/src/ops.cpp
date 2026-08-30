#include "tfm/ops.hpp"

#include <cmath>
#include <cstddef>

namespace tfm {

void linear(const Mat& x, const Mat& W, const Vec& b, Mat& y) {
  const int M = x.R, K = x.C, N = W.R;
  y = Mat(M, N);
  for (int i = 0; i < M; ++i) {
    const float* xi = x.row(i);
    for (int j = 0; j < N; ++j) {
      const float* wj = W.row(j);
      float acc = b[j];
      for (int k = 0; k < K; ++k) acc += xi[k] * wj[k];
      y.at(i, j) = acc;
    }
  }
}

void linearSlice(const Mat& x, const Mat& W, const Vec& b,
                 int slice_off, int len, Mat& y) {
  const int M = x.R, K = x.C;
  y = Mat(M, len);
  for (int i = 0; i < M; ++i) {
    const float* xi = x.row(i);
    for (int j = 0; j < len; ++j) {
      const float* wj = W.row(slice_off + j);
      float acc = b[slice_off + j];
      for (int k = 0; k < K; ++k) acc += xi[k] * wj[k];
      y.at(i, j) = acc;
    }
  }
}

void relu(Mat& x) {
  for (float& v : x.d) v = v > 0.0f ? v : 0.0f;
}

void geluExact(Mat& x) {
  const float inv_sqrt2 = 0.70710678118654752440f;  // 1/sqrt(2)
  for (float& v : x.d) {
    v = v * 0.5f * (1.0f + std::erf(v * inv_sqrt2));
  }
}

void softmaxRows(Mat& x) {
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

  Mat scores(N, N), o(N, d, 0.0f);
  for (int h = 0; h < nhead; ++h) {
    const int base = h * hd;
    // scores[i][j] = scale * sum_l q[i][base+l] * k[j][base+l]
    for (int i = 0; i < N; ++i) {
      const float* qi = q.row(i);
      for (int j = 0; j < N; ++j) {
        const float* kj = k.row(j);
        float acc = 0.0f;
        for (int l = 0; l < hd; ++l) acc += qi[base + l] * kj[base + l];
        scores.at(i, j) = acc * scale;
      }
    }
    softmaxRows(scores);
    for (int i = 0; i < N; ++i) {
      const float* si = scores.row(i);
      float* oi = o.row(i);
      for (int m = 0; m < hd; ++m) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) acc += si[j] * v.at(j, base + m);
        oi[base + m] = acc;
      }
    }
  }
  linear(o, outProjW, outProjB, out);
}

}  // namespace tfm