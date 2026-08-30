#include "tfm/layers.hpp"

#include <cmath>
#include <cstddef>

namespace tfm {

void CoordMLP::forward(const Mat& coords, Mat& out) const {
  Mat h;
  linear(coords, w0, b0, h);  // (N, pos_hidden)
  relu(h);
  linear(h, w1, b1, out);     // (N, d_model)
}

void makeCoords(int n, Mat& coords) {
  coords = Mat(n * n * n, 3);
  const float denom = (float)(n - 1);
  int idx = 0;
  for (int x = 0; x < n; ++x) {
    for (int y = 0; y < n; ++y) {
      for (int z = 0; z < n; ++z) {
        coords.at(idx, 0) = (float)x / denom;
        coords.at(idx, 1) = (float)y / denom;
        coords.at(idx, 2) = (float)z / denom;
        ++idx;
      }
    }
  }
}

void embedAndAddPos(const Mat& cellEmbed, const CoordMLP& pos,
                    const int* tokens, int n, Mat& out) {
  Mat coords;
  makeCoords(n, coords);
  Mat p;
  pos.forward(coords, p);

  const int N = n * n * n, d = cellEmbed.C;
  out = Mat(N, d);
  for (int i = 0; i < N; ++i) {
    const int t = tokens[i];
    const float* e = cellEmbed.row(t);
    const float* pp = p.row(i);
    float* oi = out.row(i);
    for (int j = 0; j < d; ++j) oi[j] = e[j] + pp[j];
  }
}

void Block::forward(const Mat& x, Mat& out) const {
  const int d = x.C;
  const float scale = 1.0f / std::sqrt((float)(d / nhead));

  Mat att;
  attention(x, nhead, scale, inProjW, inProjB, outProjW, outProjB, att);  // (N,d)
  for (int i = 0; i < att.R; ++i)
    for (int j = 0; j < att.C; ++j) att.at(i, j) += x.at(i, j);

  Mat h1;
  layerNorm(att, norm1W, norm1B, lnEps, h1);

  Mat ff;
  linear(h1, lin1W, lin1B, ff);  // (N, d_ff)
  geluExact(ff);
  Mat ff2;
  linear(ff, lin2W, lin2B, ff2);  // (N, d)
  for (int i = 0; i < ff2.R; ++i)
    for (int j = 0; j < ff2.C; ++j) ff2.at(i, j) += h1.at(i, j);

  layerNorm(ff2, norm2W, norm2B, lnEps, out);
}

}  // namespace tfm