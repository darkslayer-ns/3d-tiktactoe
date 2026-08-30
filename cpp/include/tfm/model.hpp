#pragma once

#include <string>
#include <vector>

#include "layers.hpp"

namespace tfm {

// Size-agnostic value+policy transformer, a byte-for-byte reimplementation of
// backend.ml.model.ValuePolicyTransformer in eval mode (dropout off).
struct Model {
  struct Config {
    int numLayers = 2;
    int dModel = 64;
    int nhead = 8;
    int posHidden = 32;
    int dFF = 256;
    float lnEps = 1e-5f;
  };

  Config cfg;
  Mat cellEmbed;    // (3, d)
  CoordMLP pos;
  std::vector<Block> blocks;
  Mat vh0W; Vec vh0B;          // value_head Linear(d,d)
  Vec vh2W; float vh2B;        // value_head Linear(d,1)  (1 x d)
  Vec phW; float phB;          // policy_head Linear(d,1)  (1 x d)

  // Loads the self-describing binary format written by tools/export_weights.py.
  bool load(const std::string& path, std::string* err);

  // Forward over ONE position. board: int[n^3] in {0,1,2}, mask: uint8[n^3]
  // (1 = legal/empty). Outputs value logit and n^3 policy logits (-inf where
  // masked). Thread-safe: all temporaries are stack/local.
  void forward(const int* board, const uint8_t* mask, int n,
               float& value, float* policy) const;

  int numel() const;  // total parameter count (sanity check)
};

}  // namespace tfm