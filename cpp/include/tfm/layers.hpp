#pragma once

#include "ops.hpp"
#include "tensor.hpp"

namespace tfm {

struct CoordMLP {
  Mat w0;  // (pos_hidden, 3)
  Vec b0;
  Mat w1;  // (d_model, pos_hidden)
  Vec b1;

  // pos = w1(gelu-free ReLU(w0*coord + b0)) + b1 ; exactly
  // torch PositionalEncoding.mlp: Linear(3,h)->ReLU->Linear(h,d).
  void forward(const Mat& coords, Mat& out) const;
};

// Coordinate grid for an n×n×n cube, matching torch PositionalEncoding:
//   z = y = xc = arange(n).float() / (n-1)
//   coords[i] = [xc[x], y[y], z[z]] for i = x*n*n + y*n + z
void makeCoords(int n, Mat& coords);

// x = cell_embed[tokens]  (tokens in {0,1,2}), then x = x + pos.
void embedAndAddPos(const Mat& cellEmbed, const CoordMLP& pos,
                    const int* tokens, int n, Mat& out);

struct Block {
  Mat inProjW;  // (3*d, d)
  Vec inProjB;
  Mat outProjW;  // (d, d)
  Vec outProjB;
  Mat lin1W;  // (d_ff, d)
  Vec lin1B;
  Mat lin2W;  // (d, d_ff)
  Vec lin2B;
  Vec norm1W, norm1B, norm2W, norm2B;

  int nhead = 8;
  float lnEps = 1e-5f;

  // Matches torch TransformerEncoderLayer(norm_first=False) in eval mode:
  //   x = norm1(x + attn(x));  x = norm2(x + ff(x))
  void forward(const Mat& x, Mat& out) const;
};

}  // namespace tfm