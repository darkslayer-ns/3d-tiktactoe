#pragma once

#include <string>

#include "model.hpp"

namespace tfm {

// Binary format (little-endian):
//   magic "TFM1" | cfg fields | sequence of named float32 tensors.
// Each tensor: u32 name_len, name bytes, u32 rank, u32 dims[rank], f32 data.
// Written by tools/export_weights.py.
bool loadWeights(const std::string& path, Model& model, std::string* err);

}  // namespace tfm