#pragma once

#include "tensor.hpp"

namespace tfm {

// y = x @ W^T + b.  x:(M,K)  W:(N,K) row-major  b:(N)  ->  y:(M,N)
void linear(const Mat& x, const Mat& W, const Vec& b, Mat& y);

// y[i][j] = sum_k x[i][k]*W[slice_off+j][k] + b[slice_off+j]
// for j in [0, len); W:(N,K). Used to project q/k/v from a stacked in_proj.
void linearSlice(const Mat& x, const Mat& W, const Vec& b,
                 int slice_off, int len, Mat& y);

void relu(Mat& x);

// Exact GELU, matching torch.nn.functional.gelu(x, approximate="none"):
//   x * 0.5 * (1 + erf(x / sqrt(2)))
void geluExact(Mat& x);

// In-place softmax over each row (max-subtracted for numerical stability).
void softmaxRows(Mat& x);

// Fused RowNorm like torch.nn.LayerNorm(eps): y = (x-mean)/sqrt(var+eps)*g + b.
void layerNorm(const Mat& x, const Vec& g, const Vec& b, float eps, Mat& y);

// out (1, x.C): mean of each column across rows (torch x.mean(dim=1)).
void meanRows(const Mat& x, Mat& out);

// Scaled dot-product attention (no mask, no causal), matching
// torch.nn.MultiheadAttention over q=k=v=x with need_weights=False.
void attention(const Mat& x, int nhead, float scale,
               const Mat& inProjW, const Vec& inProjB,
               const Mat& outProjW, const Vec& outProjB, Mat& out);

}  // namespace tfm