#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tfm/model.hpp"
#include "tfm/weights.hpp"

// tfm-cli <weights.bin> <n>  -- reads n^3 board tokens (0/1/2) from stdin,
// legal mask = empty cells. Prints value logit + policy logits.
int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: tfm-cli <weights.bin> <n>\n");
    return 2;
  }
  const std::string wpath = argv[1];
  const int n = std::atoi(argv[2]);
  if (n < 1 || n > 6) {
    std::fprintf(stderr, "n must be 1..6\n");
    return 2;
  }
  const int N = n * n * n;

  tfm::Model model;
  std::string err;
  if (!tfm::loadWeights(wpath, model, &err)) {
    std::fprintf(stderr, "load failed: %s\n", err.c_str());
    return 1;
  }

  std::vector<int> board(N);
  std::vector<uint8_t> mask(N);
  for (int i = 0; i < N; ++i) {
    if (std::scanf("%d", &board[i]) != 1) {
      std::fprintf(stderr, "expected %d board tokens\n", N);
      return 2;
    }
    mask[i] = board[i] == 0;
  }

  std::vector<float> policy(N);
  float value;
  model.forward(board.data(), mask.data(), n, value, policy.data());

  std::printf("value=%.6f\n", value);
  std::printf("policy:");
  for (int i = 0; i < N; ++i) std::printf(" %.6f", policy[i]);
  std::printf("\n");
  std::printf("params=%d\n", model.numel());
  return 0;
}