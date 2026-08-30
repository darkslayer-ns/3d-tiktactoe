#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "tfm/model.hpp"
#include "tfm/weights.hpp"

// Golden parity test: for each fixture in <fixtures>/fixture_<n>_<k>.bin,
// load the reference (PyTorch CPU float32) outputs and compare against the
// C++ engine. Fails (non-zero exit) if any value deviates by > valueTol or
// any finite policy logit by > policyTol.
//
// Fixture format:
//   i32 n, i32 N, i32 board[N], u8 mask[N], f32 value, f32 policy[N]

struct Fixture {
  int n, N;
  std::vector<int> board;
  std::vector<uint8_t> mask;
  float value;
  std::vector<float> policy;
};

static bool readI32(FILE* f, int& v) { return fread(&v, 4, 1, f) == 1; }

static bool loadFixture(const std::string& path, Fixture& fx) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  bool ok = readI32(f, fx.n);
  ok = ok && readI32(f, fx.N);
  fx.board.resize(fx.N);
  fx.mask.resize(fx.N);
  ok = ok && fread(fx.board.data(), 4, fx.N, f) == (size_t)fx.N;
  ok = ok && fread(fx.mask.data(), 1, fx.N, f) == (size_t)fx.N;
  ok = ok && fread(&fx.value, 4, 1, f) == 1;
  fx.policy.resize(fx.N);
  ok = ok && fread(fx.policy.data(), 4, fx.N, f) == (size_t)fx.N;
  std::fclose(f);
  return ok;
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: parity_test <weights.bin> <fixtures_dir>\n");
    return 2;
  }

  tfm::Model model;
  std::string err;
  if (!tfm::loadWeights(argv[1], model, &err)) {
    std::fprintf(stderr, "load failed: %s\n", err.c_str());
    return 1;
  }

  const double valueTol = 1e-3, policyTol = 1e-3;
  double maxV = 0.0, maxP = 0.0;
  int checked = 0, failed = 0;

  for (int n : {3, 4, 5}) {
    for (int k = 0; k < 3; ++k) {
      const std::string path =
          std::string(argv[2]) + "/fixture_" + std::to_string(n) + "_" + std::to_string(k) + ".bin";
      Fixture fx;
      if (!loadFixture(path, fx)) {
        std::fprintf(stderr, "missing fixture %s\n", path.c_str());
        return 1;
      }

      std::vector<float> gotPolicy(fx.N);
      float gotValue = 0.0f;
      model.forward(fx.board.data(), fx.mask.data(), fx.n, gotValue, gotPolicy.data());

      double dv = std::fabs((double)gotValue - fx.value);
      maxV = std::max(maxV, dv);
      double dp = 0.0;
      int finite = 0;
      for (int i = 0; i < fx.N; ++i) {
        if (std::isfinite(fx.policy[i])) {
          dp = std::max(dp, std::fabs((double)gotPolicy[i] - fx.policy[i]));
          ++finite;
        } else if (std::isfinite(gotPolicy[i])) {
          dp = std::max(dp, 1e9);  // C++ produced finite where torch had -inf
          ++finite;
        }
      }
      maxP = std::max(maxP, dp);

      const bool bad = dv > valueTol || dp > policyTol;
      failed += bad ? 1 : 0;
      ++checked;
      std::printf("n=%d k=%d  value |d|=%1.3e  policy max|d|=%1.3e  (%s)\n",
                  fx.n, k, dv, dp, bad ? "FAIL" : "ok");
    }
  }

  std::printf("\nchecked %d fixtures | max|d| value=%1.3e policy=%1.3e | "
              "tolerance value=%1.0e policy=%1.0e\n",
              checked, maxV, maxP, valueTol, policyTol);
  if (failed == 0 && maxV <= valueTol && maxP <= policyTol) {
    std::printf("PARITY: PASS\n");
    return 0;
  }
  std::printf("PARITY: FAIL (%d)\n", failed);
  return 1;
}