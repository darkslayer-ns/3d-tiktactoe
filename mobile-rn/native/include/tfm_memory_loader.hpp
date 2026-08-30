#pragma once

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "tfm/model.hpp"

// In-memory loader for the self-describing TFM1 weights format.
//
// The engine's `tfm::loadWeights(const std::string& path, ...)` reads from a
// filesystem path via fopen/fread. On mobile we embed the weights as a C array
// (see scripts/embed_weights.py -> native/include/tfm_model_data.h) and are
// forbidden from touching the filesystem at runtime, so this function parses
// the *exact same* little-endian binary format from a contiguous byte buffer.
//
// Format (mirrors cpp/src/weights.cpp):
//   magic "TFM1" | cfg: u32 numLayers, u32 dModel, u32 nhead, u32 posHidden,
//                    u32 dFF, f32 lnEps
//   then zero or more named float32 tensors:
//     u32 name_len | name bytes | u32 rank | u32 dims[rank] | f32 data[numel]

namespace tfm {

namespace tfm_mem_detail {

struct Reader {
  const unsigned char* p;
  size_t size;
  size_t off = 0;

  bool read(void* dst, size_t n) {
    if (n > size - off) return false;
    std::memcpy(dst, p + off, n);
    off += n;
    return true;
  }
  bool u32(uint32_t& v) { return read(&v, 4); }
  bool f32(float& v) { return read(&v, 4); }
};

inline bool setMat(Mat& m, int rows, int cols, const float* data) {
  m = Mat(rows, cols);
  std::memcpy(m.d.data(), data, m.numel() * sizeof(float));
  return true;
}

inline bool setVec(Vec& v, int len, const float* data) {
  v.assign(data, data + len);
  return true;
}

}  // namespace tfm_mem_detail

inline bool loadWeightsFromMemory(const unsigned char* data, size_t size,
                                  Model& model, std::string* err) {
  using namespace tfm_mem_detail;
  Reader r{data, size};

  char magic[4];
  if (!r.read(magic, 4) || std::memcmp(magic, "TFM1", 4) != 0) {
    if (err) *err = "bad magic (not a TFM1 weight file)";
    return false;
  }

  uint32_t nLayers, dModel, nhead, posHidden, dFF;
  float lnEps;
  if (!r.u32(nLayers) || !r.u32(dModel) || !r.u32(nhead) || !r.u32(posHidden) ||
      !r.u32(dFF) || !r.f32(lnEps)) {
    if (err) *err = "truncated header";
    return false;
  }
  model.cfg.numLayers = (int)nLayers;
  model.cfg.dModel = (int)dModel;
  model.cfg.nhead = (int)nhead;
  model.cfg.posHidden = (int)posHidden;
  model.cfg.dFF = (int)dFF;
  model.cfg.lnEps = lnEps;

  const int N = model.cfg.numLayers;
  model.blocks.assign(N, Block());
  for (Block& b : model.blocks) {
    b.nhead = model.cfg.nhead;
    b.lnEps = model.cfg.lnEps;
  }

  // Temporary buffer large enough for the biggest tensor (in_proj 3*d*d).
  std::vector<float> buf(3 * dModel * dModel);

  while (true) {
    uint32_t nameLen;
    if (!r.u32(nameLen)) break;  // EOF after last tensor
    if (nameLen == 0 || nameLen > 4096) {
      if (err) *err = "bad tensor name length";
      return false;
    }
    std::string name(nameLen, '\0');
    if (!r.read(name.data(), nameLen)) {
      if (err) *err = "truncated tensor name";
      return false;
    }
    uint32_t rank;
    if (!r.u32(rank) || rank == 0 || rank > 8) {
      if (err) *err = "bad tensor rank for " + name;
      return false;
    }
    std::vector<uint32_t> dims(rank);
    int64_t numel = 1;
    for (uint32_t& d : dims) {
      if (!r.u32(d)) {
        if (err) *err = "truncated dims for " + name;
        return false;
      }
      numel *= (int64_t)d;
    }
    if (numel > (int64_t)buf.size()) buf.resize((size_t)numel);
    if (!r.read(buf.data(), (size_t)numel * sizeof(float))) {
      if (err) *err = "truncated data for " + name;
      return false;
    }

    // Map each state_dict key onto the model's parameter slots (mirrors
    // cpp/src/weights.cpp exactly).
    const int d = model.cfg.dModel;
    const int h = model.cfg.posHidden;
    const int ff = model.cfg.dFF;
    const float* p = buf.data();
    if (name == "cell_embed.weight") {
      setMat(model.cellEmbed, 3, d, p);
    } else if (name == "pos.mlp.0.weight") {
      setMat(model.pos.w0, h, 3, p);
    } else if (name == "pos.mlp.0.bias") {
      setVec(model.pos.b0, h, p);
    } else if (name == "pos.mlp.2.weight") {
      setMat(model.pos.w1, d, h, p);
    } else if (name == "pos.mlp.2.bias") {
      setVec(model.pos.b1, d, p);
    } else if (name == "value_head.0.weight") {
      setMat(model.vh0W, d, d, p);
    } else if (name == "value_head.0.bias") {
      setVec(model.vh0B, d, p);
    } else if (name == "value_head.2.weight") {
      setVec(model.vh2W, d, p);
    } else if (name == "value_head.2.bias") {
      model.vh2B = p[0];
    } else if (name == "policy_head.weight") {
      setVec(model.phW, d, p);
    } else if (name == "policy_head.bias") {
      model.phB = p[0];
    } else {
      // Encoder block parameters: encoder.layers.{i}.<sub>.<tail>
      int layer = -1;
      std::string tail;
      const std::string prefix = "encoder.layers.";
      if (name.rfind(prefix, 0) == 0) {
        const size_t rest = prefix.size();
        const size_t dot = name.find('.', rest);
        if (dot != std::string::npos) {
          layer = std::atoi(name.substr(rest, dot - rest).c_str());
          tail = name.substr(dot + 1);
        }
      }
      if (layer < 0 || layer >= N) {
        if (err) *err = "unknown tensor " + name;
        return false;
      }
      Block& b = model.blocks[(size_t)layer];
      if (tail == "self_attn.in_proj_weight") setMat(b.inProjW, 3 * d, d, p);
      else if (tail == "self_attn.in_proj_bias") setVec(b.inProjB, 3 * d, p);
      else if (tail == "self_attn.out_proj.weight") setMat(b.outProjW, d, d, p);
      else if (tail == "self_attn.out_proj.bias") setVec(b.outProjB, d, p);
      else if (tail == "linear1.weight") setMat(b.lin1W, ff, d, p);
      else if (tail == "linear1.bias") setVec(b.lin1B, ff, p);
      else if (tail == "linear2.weight") setMat(b.lin2W, d, ff, p);
      else if (tail == "linear2.bias") setVec(b.lin2B, d, p);
      else if (tail == "norm1.weight") setVec(b.norm1W, d, p);
      else if (tail == "norm1.bias") setVec(b.norm1B, d, p);
      else if (tail == "norm2.weight") setVec(b.norm2W, d, p);
      else if (tail == "norm2.bias") setVec(b.norm2B, d, p);
      else {
        if (err) *err = "unknown tensor " + name;
        return false;
      }
    }
  }

  return true;
}

}  // namespace tfm