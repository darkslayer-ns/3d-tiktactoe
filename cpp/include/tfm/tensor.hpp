#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace tfm {

// Small dense matrix, row-major, float32. Fixed-shape view over a flat buffer.
struct Mat {
  int R = 0;
  int C = 0;
  std::vector<float> d;

  Mat() = default;
  Mat(int rows, int cols, float fill = 0.0f) : R(rows), C(cols), d((size_t)rows * cols, fill) {}

  inline float& at(int i, int j) { return d[(size_t)i * C + j]; }
  inline float at(int i, int j) const { return d[(size_t)i * C + j]; }
  inline float* row(int i) { return d.data() + (size_t)i * C; }
  inline const float* row(int i) const { return d.data() + (size_t)i * C; }
  inline int numel() const { return R * C; }
};

using Vec = std::vector<float>;

inline constexpr float kNegInf = -std::numeric_limits<float>::infinity();

}  // namespace tfm