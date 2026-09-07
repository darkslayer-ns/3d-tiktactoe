#include "NativeAI.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tfmengine {
namespace {

std::vector<std::vector<int>> buildLines(int n) {
  std::vector<std::vector<int>> lines;
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        if (!(dx > 0 || (dx == 0 && dy > 0) ||
              (dx == 0 && dy == 0 && dz > 0))) continue;
        for (int x = 0; x < n; ++x) {
          for (int y = 0; y < n; ++y) {
            for (int z = 0; z < n; ++z) {
              const int ex = x + dx * (n - 1);
              const int ey = y + dy * (n - 1);
              const int ez = z + dz * (n - 1);
              if (ex < 0 || ex >= n || ey < 0 || ey >= n || ez < 0 || ez >= n) continue;
              std::vector<int> line;
              for (int k = 0; k < n; ++k) {
                const int px = x + dx * k;
                const int py = y + dy * k;
                const int pz = z + dz * k;
                line.push_back(px + n * (py + n * pz));
              }
              lines.push_back(std::move(line));
            }
          }
        }
      }
    }
  }
  return lines;
}

const std::vector<std::vector<int>>& linesFor(int n) {
  static std::map<int, std::vector<std::vector<int>>> cache;
  auto it = cache.find(n);
  if (it != cache.end()) return it->second;
  return cache.emplace(n, buildLines(n)).first->second;
}

int other(int side) { return side == 1 ? 2 : 1; }

double sigmoid(double x) {
  if (x >= 0.0) {
    const double z = std::exp(-x);
    return 1.0 / (1.0 + z);
  }
  const double z = std::exp(x);
  return z / (1.0 + z);
}

}  // namespace

NativeAI::NativeAI(const tfm::Model& model) : model_(model) {}

void NativeAI::start(const NativeAIConfig& config) {
  n_ = std::max(3, std::min(6, config.n));
  humanSide_ = config.humanSide == 2 ? 2 : 1;
  difficultyName_ = config.difficulty;
  aggression_ = std::max(-1.0, std::min(1.0, config.aggression));
  adaptive_ = std::max(-1.0, std::min(1.0, config.adaptive));
  cells_.assign(static_cast<size_t>(n_ * n_ * n_), 0);
  affinity_.clear();
  for (size_t i = 0; i + 2 < config.affinity.size(); i += 3) {
    affinity_[static_cast<int>(config.affinity[i])][static_cast<int>(config.affinity[i + 1])] = config.affinity[i + 2];
  }
  for (auto& [side, row] : affinity_) {
    for (auto& [cell, value] : row) value *= 0.9;
  }
  attack_ = config.profile.size() > 0 ? config.profile[0] : 0.0;
  defend_ = config.profile.size() > 1 ? config.profile[1] : 0.0;
  neutral_ = config.profile.size() > 2 ? config.profile[2] : 0.0;
  perceptionAxis_ = config.perception.size() > 0 ? config.perception[0] : 0.0;
  perceptionFace_ = config.perception.size() > 1 ? config.perception[1] : 0.0;
  perceptionSpace_ = config.perception.size() > 2 ? config.perception[2] : 0.0;
  wins_ = config.stats.size() > 0 ? config.stats[0] : 0.0;
  losses_ = config.stats.size() > 1 ? config.stats[1] : 0.0;
  draws_ = config.stats.size() > 2 ? config.stats[2] : 0.0;
  wrongMovesUsed_ = 0;
}

void NativeAI::setBoard(const std::vector<int>& cells) {
  if (cells.size() == static_cast<size_t>(n_ * n_ * n_)) cells_ = cells;
}

NativeAI::Difficulty NativeAI::difficulty() const {
  if (difficultyName_ == "easy") {
    return {1, 3, 220, 1.0, 3, 6, 0.25, 1.1, true, false, 1.0};
  }
  if (difficultyName_ == "medium") {
    return {3, 3, 220, 0.6, 1, 1, 0.25, 0.5, false, true, 0.0};
  }
  return {4, 3, 220, 0.3, 1, 0, 0.0, 0.1, false, false, 0.0};
}

int NativeAI::winner(const std::vector<int>& cells) const {
  for (const auto& line : linesFor(n_)) {
    const int first = cells[static_cast<size_t>(line[0])];
    if (first == 0) continue;
    bool all = true;
    for (int idx : line) {
      if (cells[static_cast<size_t>(idx)] != first) {
        all = false;
        break;
      }
    }
    if (all) return first;
  }
  return 0;
}

bool NativeAI::wouldWin(std::vector<int>& cells, int side, int cell) const {
  if (cell < 0 || cell >= static_cast<int>(cells.size()) || cells[static_cast<size_t>(cell)] != 0) return false;
  cells[static_cast<size_t>(cell)] = side;
  const bool won = winner(cells) == side;
  cells[static_cast<size_t>(cell)] = 0;
  return won;
}

std::vector<int> NativeAI::winningCells(const std::vector<int>& cells, int side) const {
  std::vector<int> out;
  std::vector<int> work = cells;
  for (int i = 0; i < static_cast<int>(cells.size()); ++i) {
    if (wouldWin(work, side, i)) out.push_back(i);
  }
  return out;
}

double NativeAI::valueFor(const std::vector<int>& cells, int side) const {
  tfm::ModelSearchEngine engine(model_);
  double value = 0.0;
  std::vector<float> policy;
  engine.evalPosition(cells, side, value, policy);
  return sigmoid(value);
}

int NativeAI::greedy(const std::vector<int>& cells, int side) const {
  tfm::ModelSearchEngine engine(model_);
  double value = 0.0;
  std::vector<float> policy;
  engine.evalPosition(cells, side, value, policy);
  int best = -1;
  float bestValue = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < static_cast<int>(cells.size()); ++i) {
    if (cells[static_cast<size_t>(i)] == 0 && policy[static_cast<size_t>(i)] > bestValue) {
      bestValue = policy[static_cast<size_t>(i)];
      best = i;
    }
  }
  return best;
}

double NativeAI::rollout(std::vector<int>& cells, int toMove, int human, int pliesLeft) const {
  const int w = winner(cells);
  if (w != 0) return w == human ? 1.0 : 0.0;
  bool full = true;
  for (int c : cells) if (c == 0) { full = false; break; }
  if (full) return 0.5;

  const int foe = other(toMove);
  if (!winningCells(cells, toMove).empty()) return toMove == human ? 1.0 : 0.0;
  const auto threats = winningCells(cells, foe);
  if (threats.size() == 1) {
    cells[static_cast<size_t>(threats[0])] = toMove;
    const double v = rollout(cells, foe, human, pliesLeft - 1);
    cells[static_cast<size_t>(threats[0])] = 0;
    return v;
  }
  if (threats.size() > 1) return toMove == human ? 0.0 : 1.0;
  if (pliesLeft <= 0) {
    const double v = valueFor(cells, toMove);
    return toMove == human ? v : 1.0 - v;
  }
  const int move = greedy(cells, toMove);
  if (move < 0) return 0.5;
  cells[static_cast<size_t>(move)] = toMove;
  const double v = rollout(cells, foe, human, pliesLeft - 1);
  cells[static_cast<size_t>(move)] = 0;
  return v;
}

void NativeAI::recordAffinity(int player, int cell) {
  affinity_[player][cell] += 1.0;
}

void NativeAI::recordProfile(int player, int cell) {
  if (player != humanSide_) return;
  const int opp = other(player);
  perceptionAxis_ *= 0.95;
  perceptionFace_ *= 0.95;
  perceptionSpace_ *= 0.95;
  auto creditAxis = [&](const std::vector<int>& board, int side) {
    for (const auto& line : linesFor(n_)) {
      if (std::find(line.begin(), line.end(), cell) == line.end()) continue;
      int count = 0;
      for (int index : line) if (board[static_cast<size_t>(index)] == side) ++count;
      if (count < n_ - 1) continue;
      const int dx = std::abs((line[1] % n_) - (line[0] % n_));
      const int dy = std::abs(((line[1] / n_) % n_) - ((line[0] / n_) % n_));
      const int dz = std::abs((line[1] / (n_ * n_)) - (line[0] / (n_ * n_)));
      const int dimensions = (dx > 0) + (dy > 0) + (dz > 0);
      if (dimensions == 1) perceptionAxis_ += 1.0;
      else if (dimensions == 2) perceptionFace_ += 1.0;
      else perceptionSpace_ += 1.0;
      return;
    }
  };
  std::vector<int> work = cells_;
  const bool defend = wouldWin(work, opp, cell);
  if (defend) {
    attack_ *= 0.95; defend_ *= 0.95; neutral_ *= 0.95; defend_ += 1.0;
    creditAxis(cells_, opp);
    return;
  }
  work[static_cast<size_t>(cell)] = player;
  const bool attack = winner(work) == player || !winningCells(work, player).empty();
  attack_ *= 0.95; defend_ *= 0.95; neutral_ *= 0.95;
  if (attack) {
    attack_ += 1.0;
    creditAxis(work, player);
  } else neutral_ += 1.0;
}

void NativeAI::applyMove(int player, int cell) {
  if (cell < 0 || cell >= static_cast<int>(cells_.size()) || cells_[static_cast<size_t>(cell)] != 0) return;
  recordProfile(player, cell);
  recordAffinity(player, cell);
  cells_[static_cast<size_t>(cell)] = player;
}

int NativeAI::pickBlunder(const std::vector<tfm::ScoredMove>& scored, int best,
                          const std::vector<int>& moves) {
  if (scored.size() > 1) {
    const int maxRank = std::min(static_cast<int>(scored.size()) - 1, 3);
    std::uniform_int_distribution<int> dist(1, maxRank);
    return scored[static_cast<size_t>(dist(rng_))].move;
  }
  std::vector<int> others;
  for (int m : moves) if (m != best) others.push_back(m);
  if (others.empty()) return best;
  std::uniform_int_distribution<int> dist(0, static_cast<int>(others.size()) - 1);
  return others[static_cast<size_t>(dist(rng_))];
}

NativeAIResult NativeAI::chooseMove(int player) {
  NativeAIResult result;
  const Difficulty cfg = difficulty();
  std::vector<int> moves;
  for (int i = 0; i < static_cast<int>(cells_.size()); ++i) if (cells_[static_cast<size_t>(i)] == 0) moves.push_back(i);
  if (moves.empty()) return result;
  const auto winsNow = winningCells(cells_, player);
  if (!winsNow.empty()) {
    std::uniform_real_distribution<double> random(0.0, 1.0);
    if (cfg.suboptimalBlunder && wrongMovesUsed_ < cfg.wrongMoveBudget && random(rng_) < cfg.mistakeRate) {
      ++wrongMovesUsed_;
      std::vector<int> alternatives;
      for (int move : moves) if (move != winsNow.front()) alternatives.push_back(move);
      result.move = alternatives.empty() ? winsNow.front() : alternatives[static_cast<size_t>(random(rng_) * alternatives.size())];
      result.kind = "blunder";
    } else {
      result.move = winsNow.front();
      result.kind = "search";
    }
    result.depth = cfg.depth;
    return result;
  }
  const int opp = other(player);
  const auto blocks = winningCells(cells_, opp);
  if (blocks.size() == 1) {
    result.move = blocks[0]; result.kind = "search"; result.depth = cfg.depth; return result;
  }

  tfm::ModelSearchEngine engine(model_);
  tfm::SearchCore core(engine, n_, cells_, player, cfg.depth, cfg.topK, cfg.maxNodes, aggression_);
  result.scored = core.scored();
  if (result.scored.empty()) return result;
  const int human = other(player);
  const auto humanAffinity = affinity_.find(human);
  for (auto& scored : result.scored) {
    if (humanAffinity != affinity_.end()) {
      auto it = humanAffinity->second.find(scored.move);
      if (it != humanAffinity->second.end() && it->second >= 1.0) scored.value += 0.05 * it->second;
    }
    if (cfg.defensive > 0.0) {
      const double block = wouldWin(cells_, human, scored.move) ? 1.0 : 0.0;
      auto work = cells_;
      work[static_cast<size_t>(scored.move)] = player;
      const double attack = !winningCells(work, player).empty() ? 1.0 : 0.0;
      scored.value += 0.15 * cfg.defensive * (block - attack);
    }
  }
  std::sort(result.scored.begin(), result.scored.end(), [](const auto& a, const auto& b) { return a.value > b.value; });
  int best = result.scored.front().move;
  result.value = result.scored.front().value;
  const double rate = std::max(0.0, std::min(0.8, cfg.mistakeRate + adaptive_ * 0.3 + aggression_ * 0.1));
  const bool predicament = cfg.randomBlunder && result.value >= 0.85;
  const double effectiveRate = predicament ? std::max(rate, 0.9) : rate;
  std::uniform_real_distribution<double> random(0.0, 1.0);
  const bool withinBudget = wrongMovesUsed_ < cfg.wrongMoveBudget;
  const bool forcedWin = !winningCells(cells_, player).empty();
  bool blunder = false;
  if (withinBudget && cfg.randomBlunder && effectiveRate > 0.0 && random(rng_) < effectiveRate) blunder = true;
  if (withinBudget && cfg.suboptimalBlunder && forcedWin && random(rng_) < rate) blunder = true;
  int selected = best;
  auto sampleTop = [&](int count, double temperature) {
    count = std::min(count, static_cast<int>(result.scored.size()));
    if (count <= 1 || temperature <= 0.0) return result.scored.front().move;
    std::vector<double> weights;
    weights.reserve(static_cast<size_t>(count));
    double maxValue = result.scored.front().value;
    for (int i = 0; i < count; ++i) weights.push_back(std::exp((result.scored[static_cast<size_t>(i)].value - maxValue) / temperature));
    std::discrete_distribution<int> pick(weights.begin(), weights.end());
    return result.scored[static_cast<size_t>(pick(rng_))].move;
  };
  const int movesPlayed = static_cast<int>(cells_.size()) - static_cast<int>(moves.size());
  if (movesPlayed < cfg.entryMoves) selected = sampleTop(5, cfg.entryTemp);
  else selected = sampleTop(3, cfg.moveTemp);
  if (blunder) {
    ++wrongMovesUsed_;
    result.move = cfg.randomBlunder ? moves[static_cast<size_t>(random(rng_) * moves.size())] : pickBlunder(result.scored, best, moves);
    result.kind = "blunder";
  } else {
    result.move = selected;
  }
  result.depth = cfg.depth;
  result.line = core.predictedLine(result.move);
  return result;
}

int NativeAI::hint(int player) {
  const auto win = winningCells(cells_, player);
  if (!win.empty()) return win.front();
  const auto blocks = winningCells(cells_, other(player));
  if (blocks.size() == 1) return blocks.front();
  std::vector<int> moves;
  for (int i = 0; i < static_cast<int>(cells_.size()); ++i) if (cells_[static_cast<size_t>(i)] == 0) moves.push_back(i);
  if (moves.empty()) return -1;
  tfm::ModelSearchEngine engine(model_);
  double rootValue = 0.0; std::vector<float> rootPolicy;
  engine.evalPosition(cells_, player, rootValue, rootPolicy);
  int best = moves.front(); double bestValue = -1.0;
  const int plies = std::max(4, 8 - std::max(0, n_ - 3));
  for (int move : moves) {
    cells_[static_cast<size_t>(move)] = player;
    const double value = rollout(cells_, other(player), player, plies - 1);
    cells_[static_cast<size_t>(move)] = 0;
    if (value > bestValue + 1e-9 || (std::abs(value - bestValue) <= 1e-9 && rootPolicy[static_cast<size_t>(move)] > rootPolicy[static_cast<size_t>(best)])) {
      best = move; bestValue = value;
    }
  }
  return best;
}

NativeAIState NativeAI::endGame(int winnerSide) {
  if (winnerSide == 0) ++draws_;
  else if (winnerSide == humanSide_) ++wins_;
  else ++losses_;
  if (winnerSide != 0) {
    const int loser = other(winnerSide);
    for (auto& [cell, value] : affinity_[winnerSide]) value *= 1.25;
    for (auto& [cell, value] : affinity_[loser]) value *= 0.5;
  }
  const double total = wins_ + losses_;
  const double rate = total < 2.0 ? 0.5 : wins_ / total;
  adaptive_ = std::max(-1.0, std::min(1.0, (0.55 - rate) * 2.5));
  return state();
}

NativeAIState NativeAI::state() const {
  NativeAIState out;
  for (const auto& [side, row] : affinity_) for (const auto& [cell, value] : row) {
    if (value > 0.0) { out.affinity.push_back(side); out.affinity.push_back(cell); out.affinity.push_back(value); }
  }
  out.profile = {attack_, defend_, neutral_};
  out.perception = {perceptionAxis_, perceptionFace_, perceptionSpace_};
  out.stats = {wins_, losses_, draws_};
  out.adaptive = adaptive_;
  out.aggression = (attack_ + defend_ < 0.5) ? 0.0 : (attack_ - defend_) / (attack_ + defend_);
  return out;
}

NativeAIKnowledge NativeAI::knowledge(int humanSide) const {
  NativeAIKnowledge out;
  const NativeAIState base = state();
  out.affinity = base.affinity;
  out.profile = base.profile;
  out.perception = base.perception;
  out.stats = base.stats;
  out.adaptive = base.adaptive;
  out.aggression = base.aggression;

  const int human = humanSide == 2 ? 2 : 1;
  const int ai = other(human);
  tfm::ModelSearchEngine engine(model_);
  double humanLogit = 0.0;
  double aiLogit = 0.0;
  std::vector<float> humanPolicy;
  std::vector<float> aiPolicy;
  engine.evalPosition(cells_, human, humanLogit, humanPolicy);
  engine.evalPosition(cells_, ai, aiLogit, aiPolicy);
  out.winProbHuman = sigmoid(humanLogit);
  out.winProbAi = sigmoid(aiLogit);
  float bestPolicy = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < static_cast<int>(cells_.size()); ++i) {
    if (cells_[static_cast<size_t>(i)] == 0 && aiPolicy[static_cast<size_t>(i)] > bestPolicy) {
      bestPolicy = aiPolicy[static_cast<size_t>(i)];
      out.bestMoveIndex = i;
    }
  }

  std::vector<std::pair<int, double>> scores;
  double maxScore = -std::numeric_limits<double>::infinity();
  const auto rowIt = affinity_.find(human);
  for (int i = 0; i < static_cast<int>(cells_.size()); ++i) {
    if (cells_[static_cast<size_t>(i)] != 0) continue;
    auto work = cells_;
    work[static_cast<size_t>(i)] = human;
    double valueLogit = 0.0;
    std::vector<float> policy;
    engine.evalPosition(work, human, valueLogit, policy);
    double score = sigmoid(valueLogit);
    if (rowIt != affinity_.end()) {
      auto affinityCell = rowIt->second.find(i);
      if (affinityCell != rowIt->second.end()) score += affinityCell->second;
    }
    scores.push_back({i, score});
    maxScore = std::max(maxScore, score);
  }
  double total = 0.0;
  for (auto& score : scores) {
    score.second = std::exp(score.second - maxScore);
    total += score.second;
  }
  for (auto& score : scores) score.second /= std::max(total, 1e-12);
  std::sort(scores.begin(), scores.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
  const size_t limit = std::min<size_t>(64, scores.size());
  for (size_t i = 0; i < limit; ++i) out.predictions.push_back({scores[i].first, scores[i].second});
  return out;
}

}  // namespace tfmengine
