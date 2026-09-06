#pragma once

#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "tfm/model.hpp"
#include "tfm/search.hpp"

namespace tfmengine {

struct NativeAIConfig {
  int n = 3;
  int humanSide = 1;
  std::string difficulty = "hard";
  double aggression = 0.0;
  double adaptive = 0.0;
  std::vector<double> affinity;    // side, cell, weight triples
  std::vector<double> profile;     // attack, defend, neutral
  std::vector<double> perception;  // axis, face, space
  std::vector<double> stats;       // wins, losses, draws
};

struct NativeAIResult {
  int move = -1;
  std::string kind = "search";
  double value = 0.0;
  int depth = 0;
  std::vector<tfm::ScoredMove> scored;
  std::vector<tfm::LineStep> line;
};

struct NativeAIState {
  std::vector<double> affinity;
  std::vector<double> profile;
  std::vector<double> perception;
  std::vector<double> stats;
  double adaptive = 0.0;
  double aggression = 0.0;
};

class NativeAI {
 public:
  explicit NativeAI(const tfm::Model& model);

  void start(const NativeAIConfig& config);
  void setBoard(const std::vector<int>& cells);
  void applyMove(int player, int cell);
  NativeAIResult chooseMove(int player);
  int hint(int player);
  NativeAIState endGame(int winner);
  NativeAIState state() const;

 private:
  struct Difficulty {
    int depth;
    int topK;
    int maxNodes;
    double entryTemp;
    int entryMoves;
    int wrongMoveBudget;
    double mistakeRate;
    double moveTemp;
    bool randomBlunder;
    bool suboptimalBlunder;
    double defensive;
  };

  Difficulty difficulty() const;
  std::vector<int> winningCells(const std::vector<int>& cells, int side) const;
  int winner(const std::vector<int>& cells) const;
  bool wouldWin(std::vector<int>& cells, int side, int cell) const;
  double valueFor(const std::vector<int>& cells, int side) const;
  int greedy(const std::vector<int>& cells, int side) const;
  double rollout(std::vector<int>& cells, int toMove, int human,
                 int pliesLeft) const;
  int pickBlunder(const std::vector<tfm::ScoredMove>& scored, int best,
                  const std::vector<int>& moves);
  void recordAffinity(int player, int cell);
  void recordProfile(int player, int cell);

  const tfm::Model& model_;
  int n_ = 3;
  int humanSide_ = 1;
  std::string difficultyName_ = "hard";
  double aggression_ = 0.0;
  double adaptive_ = 0.0;
  std::vector<int> cells_;
  std::map<int, std::map<int, double>> affinity_;
  double attack_ = 0.0;
  double defend_ = 0.0;
  double neutral_ = 0.0;
  double perceptionAxis_ = 0.0;
  double perceptionFace_ = 0.0;
  double perceptionSpace_ = 0.0;
  double wins_ = 0.0;
  double losses_ = 0.0;
  double draws_ = 0.0;
  int wrongMovesUsed_ = 0;
  std::mt19937 rng_{std::random_device{}()};
};

}  // namespace tfmengine
