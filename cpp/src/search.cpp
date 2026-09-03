#include "tfm/search.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <mutex>
#include <utility>

namespace tfm {

namespace {

constexpr int kEmpty = 0;
constexpr int kP1 = 1;
constexpr int kP2 = 2;

// Abramowitz & Stegun 7.1.26 erf polynomial — same one ops.cpp uses, kept in
// double here (never used for parity-critical math; see sigmoid note below).
inline double sigmoid(double x) {
  // Two-branch stable sigmoid — byte-for-byte port of cpp_inference.sigmoid
  // / src/ai/math.ts sigmoid (both consume float32 logits widened to double).
  if (x >= 0) {
    const double z = std::exp(-x);
    return 1.0 / (1.0 + z);
  }
  const double z = std::exp(x);
  return z / (1.0 + z);
}

int argmax(const std::vector<float>& logits) {
  // First occurrence wins on ties — port of cpp_inference.argmax / math.ts.
  float best = logits[0];
  int bestI = 0;
  for (size_t i = 0; i < logits.size(); ++i) {
    if (logits[i] > best) {
      best = logits[i];
      bestI = (int)i;
    }
  }
  return bestI;
}

// Softmax at temperature 1.0 over ALL logits (occupied cells carry -inf or
// ±1e9 from the fixture encoding; exp() underflows both to exactly 0.0).
std::vector<double> softmax1(const std::vector<float>& logits) {
  double m = -std::numeric_limits<double>::infinity();
  for (float v : logits) {
    const double d = (double)v;
    if (d > m) m = d;
  }
  std::vector<double> exps(logits.size());
  double total = 0.0;
  for (size_t i = 0; i < logits.size(); ++i) {
    exps[i] = std::exp((double)logits[i] - m);
    total += exps[i];
  }
  for (double& e : exps) e /= total;
  return exps;
}

// ---------------------------------------------------------------------------
// Board logic — port of src/game/board.ts (== backend/game/board.py).
// ---------------------------------------------------------------------------

// All winning lines for size n, as flat cell-index lists, in the exact
// enumeration order of Board.buildLines (canonical directions, then the
// (start-cell, direction) key sort). Cached per size.
const std::vector<std::vector<int>>& linesFor(int n) {
  static std::mutex mu;
  static std::map<int, std::vector<std::vector<int>>> cache;
  std::lock_guard<std::mutex> lock(mu);
  auto it = cache.find(n);
  if (it != cache.end()) return it->second;

  // canonical primitive direction vectors (no opposites, zero excluded)
  std::vector<std::array<int, 3>> dirs;
  for (int a = -1; a <= 1; ++a) {
    for (int b = -1; b <= 1; ++b) {
      for (int c = -1; c <= 1; ++c) {
        if (a == 0 && b == 0 && c == 0) continue;
        if (a > 0 || (a == 0 && b > 0) || (a == 0 && b == 0 && c > 0)) {
          dirs.push_back({a, b, c});
        }
      }
    }
  }

  struct Entry {
    int64_t key;
    std::vector<int> line;
  };
  std::vector<Entry> entries;
  int dirIdx = 0;
  for (const auto& [dx, dy, dz] : dirs) {
    const int xLo = dx == 1 ? 0 : dx == -1 ? n - 1 : 0;
    const int xHi = dx == 0 ? n - 1 : xLo;
    const int yLo = dy == 1 ? 0 : dy == -1 ? n - 1 : 0;
    const int yHi = dy == 0 ? n - 1 : yLo;
    const int zLo = dz == 1 ? 0 : dz == -1 ? n - 1 : 0;
    const int zHi = dz == 0 ? n - 1 : zLo;
    for (int x = xLo; x <= xHi; ++x) {
      for (int y = yLo; y <= yHi; ++y) {
        for (int z = zLo; z <= zHi; ++z) {
          std::vector<int> line((size_t)n);
          for (int k = 0; k < n; ++k) {
            const int lx = x + dx * k, ly = y + dy * k, lz = z + dz * k;
            line[(size_t)k] = lx + n * (ly + n * lz);
          }
          entries.push_back({((int64_t)(x * n + y) * n + z) * (int64_t)dirs.size() + dirIdx,
                             std::move(line)});
        }
      }
    }
    ++dirIdx;
  }
  std::stable_sort(entries.begin(), entries.end(),
                   [](const Entry& a, const Entry& b) { return a.key < b.key; });
  std::vector<std::vector<int>> lines;
  lines.reserve(entries.size());
  for (auto& e : entries) lines.push_back(std::move(e.line));
  cache[n] = std::move(lines);
  return cache[n];
}

int findWinner(const std::vector<int>& cells,
               const std::vector<std::vector<int>>& lines) {
  for (const auto& line : lines) {
    const int first = cells[(size_t)line[0]];
    if (first == kEmpty) continue;
    bool all = true;
    for (size_t i = 1; i < line.size(); ++i) {
      if (cells[(size_t)line[i]] != first) {
        all = false;
        break;
      }
    }
    if (all) return first;
  }
  return kEmpty;
}

bool boardFull(const std::vector<int>& cells) {
  for (int c : cells)
    if (c == kEmpty) return false;
  return true;
}

void normalizeBoard(const std::vector<int>& cells, int side, int N,
                    std::vector<int>& norm, std::vector<uint8_t>& mask) {
  norm.resize((size_t)N);
  mask.resize((size_t)N);
  for (int i = 0; i < N; ++i) {
    const int c = cells[(size_t)i];
    norm[(size_t)i] = c == 0 ? 0 : (c == side ? 1 : 2);
    mask[(size_t)i] = c == 0 ? 1 : 0;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// SearchEngine seam
// ---------------------------------------------------------------------------

void SearchEngine::evalPositions(const std::vector<int>& boards,
                                 const std::vector<uint8_t>& masks, int n,
                                 int count, int side,
                                 const std::vector<int>* rawCells,
                                 std::vector<float>& valueLogits) {
  // Fallback: one evalPosition per board — values identical to a real batch
  // (Model::forwardBatch computes each board with the same scalar forward).
  const int N = n * n * n;
  valueLogits.resize((size_t)count);
  for (int i = 0; i < count; ++i) {
    double v = 0.0;
    std::vector<float> policy;
    if (rawCells != nullptr) {
      evalPosition(std::vector<int>(rawCells->begin() + (size_t)i * N,
                                    rawCells->begin() + (size_t)(i + 1) * N),
                   side, v, policy);
    } else {
      // No raw cells supplied: un-normalize is impossible, so re-normalize
      // from the provided tokens is skipped — callers must pass rawCells to
      // use the fallback. Batch-capable engines override this method.
      (void)boards;
      (void)masks;
      valueLogits[(size_t)i] = std::numeric_limits<float>::quiet_NaN();
    }
    valueLogits[(size_t)i] = (float)v;
  }
}

void ModelSearchEngine::evalPosition(const std::vector<int>& cells, int player,
                                     double& valueLogit,
                                     std::vector<float>& policy) {
  const int n = (int)std::lround(std::cbrt((double)cells.size()));
  const int N = n * n * n;
  std::vector<int> norm;
  std::vector<uint8_t> mask;
  normalizeBoard(cells, player, N, norm, mask);
  policy.resize((size_t)N);
  float v = 0.0f;
  model_.forward(norm.data(), mask.data(), n, v, policy.data());
  valueLogit = (double)v;
}

void ModelSearchEngine::evalPositions(const std::vector<int>& boards,
                                      const std::vector<uint8_t>& masks, int n,
                                      int count, int side,
                                      const std::vector<int>* rawCells,
                                      std::vector<float>& valueLogits) {
  (void)side;
  (void)rawCells;
  const int N = n * n * n;
  valueLogits.resize((size_t)count);
  if (count <= 0) return;
  std::vector<float> policies((size_t)count * N);  // unused by the search
  model_.forwardBatch(n, count, boards.data(), masks.data(), valueLogits.data(),
                      policies.data());
}

// ---------------------------------------------------------------------------
// SearchCore — the expectimax port
// ---------------------------------------------------------------------------

SearchCore::SearchCore(SearchEngine& engine, int n, const std::vector<int>& cells,
                       int ai, int depth, int topK, int maxNodes,
                       double aggression)
    : engine_(engine),
      lines_(linesFor(n)),
      cells_(cells),
      n_(n),
      N_(n * n * n),
      ai_(ai),
      opp_(ai == kP1 ? kP2 : kP1),
      depth_(depth),
      topK_(topK),
      maxNodes_(maxNodes),
      aggression_(aggression) {
  // reset mutable search state
  nodes_ = 0;
  evalCalls_ = 0;
}

void SearchCore::evaluate(int side, double& value, int& greedyMove) {
  double valueLogit = 0.0;
  std::vector<float> policy;
  engine_.evalPosition(cells_, side, valueLogit, policy);
  ++evalCalls_;
  value = sigmoid(valueLogit);
  greedyMove = argmax(policy);
}

double SearchCore::forwardValue(int side) {
  double valueLogit = 0.0;
  std::vector<float> policy;
  engine_.evalPosition(cells_, side, valueLogit, policy);
  ++evalCalls_;
  return sigmoid(valueLogit);
}

double SearchCore::valueFor(int side) { return forwardValue(side); }

int SearchCore::greedy(int side) {
  double valueLogit = 0.0;
  std::vector<float> policy;
  engine_.evalPosition(cells_, side, valueLogit, policy);
  ++evalCalls_;
  return argmax(policy);
}

bool SearchCore::terminal(double& out) {
  const int w = findWinner(cells_, lines_);
  if (w != kEmpty) {
    out = w == ai_ ? 1.0 : 0.0;
    return true;
  }
  if (boardFull(cells_)) {
    out = 0.5;
    return true;
  }
  return false;
}

bool SearchCore::wouldWin(int side, int cell) {
  if (cells_[(size_t)cell] != kEmpty) return false;
  cells_[(size_t)cell] = side;
  const bool w = findWinner(cells_, lines_) == side;
  cells_[(size_t)cell] = kEmpty;
  return w;
}

bool SearchCore::hasWinInOne(int side) {
  for (int i = 0; i < N_; ++i) {
    if (cells_[(size_t)i] == kEmpty && wouldWin(side, i)) return true;
  }
  return false;
}

double SearchCore::attackIndicator(int side, int r) {
  if (wouldWin(side, r)) return 1.0;
  const int prev = cells_[(size_t)r];
  cells_[(size_t)r] = side;
  const bool threat = hasWinInOne(side);
  cells_[(size_t)r] = prev;
  return threat ? 1.0 : 0.0;
}

double SearchCore::defendIndicator(int r) {
  return wouldWin(ai_, r) ? 1.0 : 0.0;
}

std::vector<std::pair<int, double>> SearchCore::likelyMoves(int side, int topK) {
  double valueLogit = 0.0;
  std::vector<float> policy;
  engine_.evalPosition(cells_, side, valueLogit, policy);
  ++evalCalls_;
  const std::vector<double> probs = softmax1(policy);
  std::vector<std::pair<int, double>> scored;
  for (int i = 0; i < N_; ++i) {
    if (cells_[(size_t)i] == kEmpty) {
      scored.emplace_back(i, probs[(size_t)i]);
    }
  }
  // Stable, descending by probability — matches Python list.sort (stable)
  // and JS Array.prototype.sort (spec-stable).
  std::stable_sort(scored.begin(), scored.end(),
                   [](const std::pair<int, double>& a,
                      const std::pair<int, double>& b) {
                     return a.second > b.second;
                   });
  if ((int)scored.size() > topK) scored.resize((size_t)topK);
  return scored;
}

std::vector<std::pair<int, double>> SearchCore::styleWeighted(
    std::vector<std::pair<int, double>> dist) {
  if (std::fabs(aggression_) < 0.2 || dist.size() < 2) return dist;
  const double a = aggression_;
  const double damp = 0.7;
  for (auto& [r, pr] : dist) {
    const double att = attackIndicator(opp_, r);
    const double def = defendIndicator(r);
    pr = std::max(0.02, pr * (1.0 + a * damp * (att - def)));
  }
  return dist;
}

double SearchCore::opponentLeaves(
    const std::vector<std::pair<int, double>>& dist, double total) {
  // Depth-1 leaf level: each reply's value is an independent forward, so they
  // are collected and evaluated in ONE batched call (port of TS
  // _opponentLeaves; Python evaluates the same positions sequentially —
  // values are identical, only the call granularity differs).
  std::vector<double> rv((size_t)N_, 0.0);
  std::vector<int> pendingR;
  std::vector<std::vector<int>> pendingCells;
  for (const auto& [r, pr] : dist) {
    (void)pr;
    cells_[(size_t)r] = opp_;
    double t;
    if (terminal(t)) {
      rv[(size_t)r] = t;
    } else {
      pendingR.push_back(r);
      pendingCells.push_back(cells_);
    }
    cells_[(size_t)r] = kEmpty;
  }

  if (!pendingCells.empty()) {
    const int count = (int)pendingCells.size();
    std::vector<int> boards((size_t)count * N_);
    std::vector<uint8_t> masks((size_t)count * N_);
    std::vector<int> raw((size_t)count * N_);
    for (int b = 0; b < count; ++b) {
      for (int i = 0; i < N_; ++i) {
        const int c = pendingCells[(size_t)b][(size_t)i];
        boards[(size_t)b * N_ + i] = c == 0 ? 0 : (c == ai_ ? 1 : 2);
        masks[(size_t)b * N_ + i] = c == 0 ? 1 : 0;
        raw[(size_t)b * N_ + i] = c;
      }
    }
    std::vector<float> valueLogits;
    engine_.evalPositions(boards, masks, n_, count, ai_, &raw, valueLogits);
    evalCalls_ += count;
    for (int b = 0; b < count; ++b) {
      rv[(size_t)pendingR[(size_t)b]] = sigmoid((double)valueLogits[(size_t)b]);
    }
  }

  double exp = 0.0;
  for (const auto& [r, pr] : dist) {
    exp += (pr / total) * rv[(size_t)r];
  }
  return exp;
}

double SearchCore::opponentExpected(int depth, int topK) {
  nodes_ += 1;
  if (nodes_ > maxNodes_) {
    double v;
    int g;
    evaluate(ai_, v, g);
    return v;
  }

  auto dist = likelyMoves(opp_, std::max(1, topK));
  if (dist.empty()) {
    double v;
    int g;
    evaluate(ai_, v, g);
    return v;
  }

  auto replies = styleWeighted(std::move(dist));
  double total = 0.0;
  for (const auto& [r, pr] : replies) total += pr;

  if (depth <= 1) {
    return opponentLeaves(replies, total);
  }

  double exp = 0.0;
  const int nextTopK = std::max(1, topK - 1);
  for (const auto& [r, pr] : replies) {
    const double w = pr / total;
    cells_[(size_t)r] = opp_;
    double t;
    double rv;
    if (terminal(t)) {
      rv = t;
    } else {
      // AI answers greedily; the value here is unused (Python discards it in
      // the depth>1 branch too — only the argmax move matters).
      double v;
      int a;
      evaluate(ai_, v, a);
      cells_[(size_t)a] = ai_;
      rv = opponentExpected(depth - 1, nextTopK);
      cells_[(size_t)a] = kEmpty;
    }
    exp += w * rv;
    cells_[(size_t)r] = kEmpty;
  }
  return exp;
}

double SearchCore::evalMove(int m, int depth) {
  nodes_ += 1;
  if (nodes_ > maxNodes_) {
    // Budget exhausted: value the root as-is from the opponent's perspective
    // (m is NOT applied — mirrors Python/TS exactly, including their quirk
    // of clearing the unapplied cell).
    return 1.0 - valueFor(opp_);
  }
  cells_[(size_t)m] = ai_;
  double t;
  if (terminal(t)) {
    cells_[(size_t)m] = kEmpty;
    return t;
  }
  if (depth <= 1) {
    const double v = 1.0 - valueFor(opp_);
    cells_[(size_t)m] = kEmpty;
    return v;
  }
  const double v = opponentExpected(depth, topK_);
  cells_[(size_t)m] = kEmpty;
  return v;
}

std::vector<ScoredMove> SearchCore::scored() {
  std::vector<ScoredMove> out;
  for (int m = 0; m < N_; ++m) {
    if (cells_[(size_t)m] == kEmpty) {
      out.push_back({m, evalMove(m, depth_)});
    }
  }
  return out;
}

std::vector<LineStep> SearchCore::predictedLine(int chosen) {
  std::vector<LineStep> line;
  line.push_back({ai_, chosen});
  cells_[(size_t)chosen] = ai_;
  int player = opp_;
  int plies = 0;
  const int maxPlies = depth_ * 2;
  while (plies < maxPlies) {
    // over? (winner or full board)
    const int w = findWinner(cells_, lines_);
    if (w != kEmpty || boardFull(cells_)) break;
    int idx;
    if (player == ai_) {
      idx = greedy(player);
    } else {
      auto dist = likelyMoves(player, 1);
      if (dist.empty()) break;
      idx = dist[0].first;
    }
    line.push_back({player, idx});
    cells_[(size_t)idx] = player;
    player = player == kP1 ? kP2 : kP1;
    ++plies;
  }
  cells_[(size_t)chosen] = kEmpty;
  return line;
}

// ---------------------------------------------------------------------------
// One-shot production entry points
// ---------------------------------------------------------------------------

std::vector<ScoredMove> searchScored(const Model& model,
                                     const std::vector<int>& cells, int n, int ai,
                                     int depth, int topK, int maxNodes,
                                     double aggression) {
  ModelSearchEngine engine(model);
  SearchCore core(engine, n, cells, ai, depth, topK, maxNodes, aggression);
  return core.scored();
}

std::vector<LineStep> predictedLine(const Model& model,
                                    const std::vector<int>& cells, int n, int ai,
                                    int chosen, int depth) {
  ModelSearchEngine engine(model);
  SearchCore core(engine, n, cells, ai, depth, 3, 0, 0.0);
  return core.predictedLine(chosen);
}

bool winInOne(int n, const std::vector<int>& cells, int side) {
  const auto& lines = linesFor(n);
  std::vector<int> work = cells;
  const int N = n * n * n;
  for (int m = 0; m < N; ++m) {
    if (work[(size_t)m] != kEmpty) continue;
    work[(size_t)m] = side;
    if (findWinner(work, lines) == side) return true;
    work[(size_t)m] = kEmpty;
  }
  return false;
}

}  // namespace tfm
