#include "tfm/model.hpp"

#include <algorithm>
#include <cstddef>
#include <thread>
#include <vector>

namespace tfm {

namespace {

void valueHeadForward(const Model& m, const Mat& pooled, float& out) {
  Mat h0;
  linear(pooled, m.vh0W, m.vh0B, h0);
  relu(h0);
  // h0 is (1, d); vh2W is a Vec of length d
  float acc = m.vh2B;
  for (int j = 0; j < m.cfg.dModel; ++j) acc += h0.at(0, j) * m.vh2W[j];
  out = acc;
}

void policyHeadForward(const Model& m, const Mat& x, float* policy) {
  const int N = x.R;
  for (int i = 0; i < N; ++i) {
    const float* xi = x.row(i);
    float acc = m.phB;
    for (int j = 0; j < m.cfg.dModel; ++j) acc += xi[j] * m.phW[j];
    policy[i] = acc;
  }
}

}  // namespace

void Model::forward(const int* board, const uint8_t* mask, int n,
                    float& value, float* policy) const {
  Mat x;
  embedAndAddPos(cellEmbed, pos, board, n, x);  // (N, d)

  for (const Block& blk : blocks) {
    Mat y;
    blk.forward(x, y);
    x = std::move(y);
  }

  Mat pooled;
  meanRows(x, pooled);            // (1, d)
  valueHeadForward(*this, pooled, value);
  policyHeadForward(*this, x, policy);

  const int N = n * n * n;
  for (int i = 0; i < N; ++i)
    if (!mask[i]) policy[i] = kNegInf;
}

void Model::forwardBatch(int n, int count, const int* boards,
                         const uint8_t* masks, float* values,
                         float* policies) const {
  if (count <= 0) return;
  const int N = n * n * n;
#if defined(_OPENMP)
  // Parallel over the batch. `attention`'s inner head loop is nested inside
  // this region, and OpenMP nested parallelism is off by default, so each
  // worker runs a serial forward — no oversubscription.
#pragma omp parallel for schedule(dynamic)
  for (int i = 0; i < count; ++i) {
    forward(boards + (size_t)i * N, masks + (size_t)i * N, n, values[i],
            policies + (size_t)i * N);
  }
#else
  const unsigned hw = std::thread::hardware_concurrency();
  const int nThreads = std::min(count, hw > 0 ? (int)hw : 2);
  const int chunk = (count + nThreads - 1) / nThreads;

  auto worker = [&](int b0, int b1) {
    for (int i = b0; i < b1; ++i) {
      forward(boards + (size_t)i * N, masks + (size_t)i * N, n, values[i],
              policies + (size_t)i * N);
    }
  };

  std::vector<std::thread> pool;
  pool.reserve((size_t)nThreads - 1);
  int b0 = 0;
  for (int t = 0; t < nThreads; ++t) {
    const int b1 = std::min(count, b0 + chunk);
    if (b1 <= b0) break;
    if (t < nThreads - 1) pool.emplace_back(worker, b0, b1);
    else worker(b0, b1);  // run the last chunk on the calling thread
    b0 = b1;
  }
  for (auto& th : pool) th.join();
#endif
}

int Model::numel() const {
  int64_t total = 0;
  auto add = [&](const Mat& m) { total += m.numel(); };
  auto addv = [&](const Vec& v) { total += (int64_t)v.size(); };
  add(cellEmbed);
  add(pos.w0); addv(pos.b0); add(pos.w1); addv(pos.b1);
  for (const Block& b : blocks) {
    add(b.inProjW); addv(b.inProjB); add(b.outProjW); addv(b.outProjB);
    add(b.lin1W); addv(b.lin1B); add(b.lin2W); addv(b.lin2B);
    addv(b.norm1W); addv(b.norm1B); addv(b.norm2W); addv(b.norm2B);
  }
  add(vh0W); addv(vh0B); addv(vh2W); addv(phW);
  total += 2;  // scalar biases: value_head.2.bias, policy_head.bias
  return (int)total;
}

}  // namespace tfm