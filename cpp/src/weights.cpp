#include "tfm/weights.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace tfm {

namespace {

bool readExact(FILE* f, void* dst, size_t n) {
  return fread(dst, 1, n, f) == n;
}

bool readU32(FILE* f, uint32_t& v) { return readExact(f, &v, 4); }
bool readF32(FILE* f, float& v) { return readExact(f, &v, 4); }

bool setMat(Mat& m, int rows, int cols, const float* data) {
  m = Mat(rows, cols);
  std::memcpy(m.d.data(), data, m.numel() * sizeof(float));
  return true;
}

bool setVec(Vec& v, int len, const float* data) {
  v.assign(data, data + len);
  return true;
}

}  // namespace

bool loadWeights(const std::string& path, Model& model, std::string* err) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    if (err) *err = "cannot open " + path;
    return false;
  }

  char magic[4];
  if (!readExact(f, magic, 4) || std::memcmp(magic, "TFM1", 4) != 0) {
    if (err) *err = "bad magic (not a TFM1 weight file)";
    std::fclose(f);
    return false;
  }

  uint32_t nLayers, dModel, nhead, posHidden, dFF;
  float lnEps;
  if (!readU32(f, nLayers) || !readU32(f, dModel) || !readU32(f, nhead) ||
      !readU32(f, posHidden) || !readU32(f, dFF) || !readF32(f, lnEps)) {
    if (err) *err = "truncated header";
    std::fclose(f);
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
    if (!readU32(f, nameLen)) break;  // EOF after last tensor
    if (nameLen == 0 || nameLen > 4096) {
      if (err) *err = "bad tensor name length";
      std::fclose(f);
      return false;
    }
    std::string name(nameLen, '\0');
    if (!readExact(f, name.data(), nameLen)) {
      if (err) *err = "truncated tensor name";
      std::fclose(f);
      return false;
    }
    uint32_t rank;
    if (!readU32(f, rank) || rank == 0 || rank > 8) {
      if (err) *err = "bad tensor rank for " + name;
      std::fclose(f);
      return false;
    }
    std::vector<uint32_t> dims(rank);
    int64_t numel = 1;
    for (uint32_t& d : dims) {
      if (!readU32(f, d)) {
        if (err) *err = "truncated dims for " + name;
        std::fclose(f);
        return false;
      }
      numel *= (int64_t)d;
    }
    if (numel > (int64_t)buf.size()) buf.resize(numel);
    if (!readExact(f, buf.data(), numel * sizeof(float))) {
      if (err) *err = "truncated data for " + name;
      std::fclose(f);
      return false;
    }

    // Map each state_dict key onto the model's parameter slots.
    const int d = model.cfg.dModel;
    const int h = model.cfg.posHidden;
    const int ff = model.cfg.dFF;
    const float* p = buf.data();
    bool known = true;
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
        std::fclose(f);
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
        std::fclose(f);
        return false;
      }
      (void)known;
    }
  }

  std::fclose(f);
  return true;
}

}  // namespace tfm