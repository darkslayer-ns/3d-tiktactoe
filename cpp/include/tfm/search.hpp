#pragma once

#include <cstdint>
#include <vector>

#include "model.hpp"

namespace tfm {

// One (move, expected-value) pair from the lookahead search.
struct ScoredMove {
  int move;
  double value;
};

// One step of the predicted future line (telemetry).
struct LineStep {
  int player;
  int index;
};

/**
 * Injectable evaluation seam — the C++ mirror of the TS EvalEngine /
 * Python eval_position protocol. Implementations receive the RAW board
 * cells ({0,1,2}, side-to-move NOT normalized) and must normalize
 * internally, returning the value logit and the masked policy logits
 * (-inf on occupied cells).
 */
class SearchEngine {
 public:
  virtual ~SearchEngine() = default;

  virtual void evalPosition(const std::vector<int>& cells, int player,
                            double& valueLogit, std::vector<float>& policy) = 0;

  /**
   * Batched leaf evaluation (optional optimization). boards/masks are flat
   * concatenated count*n^3 slices of ALREADY-NORMALIZED tokens + legality
   * masks (same contract as Model::forwardBatch). Default implementation
   * falls back to count sequential evalPosition calls on raw cells, which
   * produces identical values.
   */
  virtual void evalPositions(const std::vector<int>& boards,
                             const std::vector<uint8_t>& masks, int n, int count,
                             int side, const std::vector<int>* rawCells,
                             std::vector<float>& valueLogits);
};

/** SearchEngine over the real embedded transformer (normalize + forward). */
class ModelSearchEngine : public SearchEngine {
 public:
  explicit ModelSearchEngine(const Model& model) : model_(model) {}

  void evalPosition(const std::vector<int>& cells, int player,
                    double& valueLogit, std::vector<float>& policy) override;

  void evalPositions(const std::vector<int>& boards,
                     const std::vector<uint8_t>& masks, int n, int count,
                     int side, const std::vector<int>* rawCells,
                     std::vector<float>& valueLogits) override;

 private:
  const Model& model_;
};

/**
 * Native port of the LookaheadMover search core (backend/ml/model_agent.py
 * and mobile-rn/src/ai/mover.ts — the two are value-identical):
 *
 *   scored()         — _scored: expectimax over every legal move
 *   predictedLine()  — _predictedLine: greedy/likely continuation
 *
 * Everything here is RNG-FREE: difficulty blunders, temperatures and
 * sampling stay in the caller. Node budget, depth, topK and style
 * aggression are constructor parameters, so the caller (TS) stays the
 * configuration authority.
 */
class SearchCore {
 public:
  SearchCore(SearchEngine& engine, int n, const std::vector<int>& cells,
             int ai, int depth, int topK, int maxNodes, double aggression);

  /** (move, value) for every legal cell, ascending index order. */
  std::vector<ScoredMove> scored();

  /** Best-guess future line if `ai` plays `chosen` (telemetry). */
  std::vector<LineStep> predictedLine(int chosen);

  long nodes() const { return nodes_; }
  long evalCalls() const { return evalCalls_; }

 private:
  double forwardValue(int side);  // sigmoid(value) for side-to-move
  void evaluate(int side, double& value, int& greedy);
  int greedy(int side);
  double valueFor(int side);

  bool terminal(double& out);
  bool wouldWin(int side, int cell);
  bool hasWinInOne(int side);
  double attackIndicator(int side, int r);
  double defendIndicator(int r);

  std::vector<std::pair<int, double>> likelyMoves(int side, int topK);
  std::vector<std::pair<int, double>> styleWeighted(
      std::vector<std::pair<int, double>> dist);
  double opponentLeaves(const std::vector<std::pair<int, double>>& dist,
                        double total);
  double opponentExpected(int depth, int topK);
  double evalMove(int m, int depth);

  // ---- state ----------------------------------------------------------
  SearchEngine& engine_;
  const std::vector<std::vector<int>>& lines_;
  std::vector<int> cells_;
  int n_, N_, ai_, opp_;
  int depth_, topK_, maxNodes_;
  double aggression_;
  long nodes_ = 0;
  long evalCalls_ = 0;
};

/** One-shot production entry points (used by the JSI layer). */
std::vector<ScoredMove> searchScored(const Model& model,
                                     const std::vector<int>& cells, int n, int ai,
                                     int depth, int topK, int maxNodes,
                                     double aggression);
std::vector<LineStep> predictedLine(const Model& model,
                                    const std::vector<int>& cells, int n, int ai,
                                    int chosen, int depth);

/** True if `side` has an immediate winning move on `cells` — port of
 *  profile.ts hasWinInOne / the forced checks in _strongMove. */
bool winInOne(int n, const std::vector<int>& cells, int side);

}  // namespace tfm
