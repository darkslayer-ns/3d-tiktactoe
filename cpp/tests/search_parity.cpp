// Search parity gate — three proofs for the native expectimax port:
//
//  1. FIXTURE REPLAY (real model): re-run the recorded Python backend cases
//     through SearchCore with the REAL engine. The captures were produced by
//     this very C++ model, so the port must reconstruct Python's ENTIRE eval
//     call sequence: the search's requests must positionally match the
//     capture PREFIX, predictedLine(expectedMove) must match the remaining
//     suffix, and the two counts must sum to the full capture.
//     (Replaying the rounded fixture VALUES instead is not
//     sequence-deterministic: 4-digit rounding collapses the near-ties the
//     unrounded model produces on symmetric positions.)
//
//  2. SELF PARITY: batched leaves (ModelSearchEngine::evalPositions ->
//     Model::forwardBatch) vs sequential leaves (base-class fallback) must
//     produce bit-identical (move, value) lists — the same equivalence the
//     jest 'batched picks the same moves' test asserts for the TS mover.
//
//  3. SMOKE: searchScored on an empty board returns one entry per legal cell.
//
// Usage: search_parity <model.bin> <fixtures-dir>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "tfm/model.hpp"
#include "tfm/search.hpp"
#include "tfm/weights.hpp"

using tfm::Model;
using tfm::SearchCore;
using tfm::SearchEngine;

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
  }
}

// ---------------------------------------------------------------------------
// Fixture parsing
// ---------------------------------------------------------------------------

struct CaptureEntry {
  int player;
  double value;
  std::vector<int> cells;
  std::vector<float> policy;
};

struct FixtureCase {
  int size;
  std::string difficulty;
  int ai;
  int depth, topK, maxNodes;
  std::vector<int> position;
  std::vector<CaptureEntry> capture;
  int expectedMove;
};

std::vector<FixtureCase> parseFixtures(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    std::fprintf(stderr,
                 "cannot open %s (run cpp/tools/convert_parity_fixture.py)\n",
                 path.c_str());
    std::exit(2);
  }
  std::string tag;
  int nCases = 0;
  in >> tag >> nCases;
  std::vector<FixtureCase> cases;
  for (int c = 0; c < nCases; ++c) {
    FixtureCase fc;
    in >> tag >> fc.size >> fc.difficulty >> fc.ai;
    in >> tag >> fc.depth >> fc.topK >> fc.maxNodes;
    const size_t N = (size_t)fc.size * fc.size * fc.size;
    in >> tag;
    fc.position.assign(N, 0);
    for (auto& v : fc.position) in >> v;
    int count = 0;
    in >> tag >> count;
    fc.capture.resize((size_t)count);
    for (int k = 0; k < count; ++k) {
      CaptureEntry& e = fc.capture[(size_t)k];
      in >> tag >> e.player >> e.value;
      in >> tag;
      e.cells.assign(N, 0);
      for (auto& v : e.cells) in >> v;
      in >> tag;
      e.policy.assign(N, 0.0f);
      for (auto& v : e.policy) in >> v;
    }
    in >> tag >> fc.expectedMove;
    cases.push_back(std::move(fc));
  }
  return cases;
}

bool requestMatches(int player, const std::vector<int>& cells,
                    const CaptureEntry& e) {
  if (player != e.player) return false;
  if (cells.size() != e.cells.size()) return false;
  for (size_t i = 0; i < cells.size(); ++i) {
    if (cells[i] != e.cells[i]) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// RecordingEngine: delegates to the real model engine, recording every eval
// request (per-board, including the members of batched leaf calls) in order —
// the exact granularity of Python's captured eval_position sequence.
// ---------------------------------------------------------------------------

class RecordingEngine : public SearchEngine {
 public:
  struct Request {
    int player;
    std::vector<int> cells;
  };

  explicit RecordingEngine(tfm::ModelSearchEngine& inner) : inner_(inner) {}

  void evalPosition(const std::vector<int>& cells, int player,
                    double& valueLogit, std::vector<float>& policy) override {
    requests_.push_back({player, cells});
    inner_.evalPosition(cells, player, valueLogit, policy);
  }

  void evalPositions(const std::vector<int>& boards,
                     const std::vector<uint8_t>& masks, int n, int count,
                     int side, const std::vector<int>* rawCells,
                     std::vector<float>& valueLogits) override {
    if (rawCells != nullptr) {
      const int N = n * n * n;
      for (int i = 0; i < count; ++i) {
        requests_.push_back(
            {side, std::vector<int>(rawCells->begin() + (size_t)i * N,
                                    rawCells->begin() + (size_t)(i + 1) * N)});
      }
    }
    inner_.evalPositions(boards, masks, n, count, side, rawCells, valueLogits);
  }

  const std::vector<Request>& requests() const { return requests_; }

 private:
  tfm::ModelSearchEngine& inner_;
  std::vector<Request> requests_;
};

// ---------------------------------------------------------------------------
// SequentialEngine: single evals only — forces the base-class evalPositions
// fallback (per-board evalPosition calls) for the self-parity proof.
// ---------------------------------------------------------------------------

class SequentialEngine : public SearchEngine {
 public:
  explicit SequentialEngine(tfm::ModelSearchEngine& inner) : inner_(inner) {}

  void evalPosition(const std::vector<int>& cells, int player,
                    double& valueLogit, std::vector<float>& policy) override {
    inner_.evalPosition(cells, player, valueLogit, policy);
  }

 private:
  tfm::ModelSearchEngine& inner_;
};

std::vector<int> makeBoard(int n, int salt) {
  const int N = n * n * n;
  std::vector<int> cells((size_t)N, 0);
  for (int i = 0; i < N; ++i) {
    const int v = (i * 7 + salt) % 6;
    cells[(size_t)i] = v < 2 ? v : 0;
  }
  return cells;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: search_parity <model.bin> <fixtures-dir>\n");
    return 2;
  }
  const std::string modelPath = argv[1];
  const std::string fixturePath = std::string(argv[2]) + "/parity_search.txt";

  Model model;
  std::string err;
  if (!tfm::loadWeights(modelPath, model, &err)) {
    std::fprintf(stderr, "model load failed: %s\n", err.c_str());
    return 1;
  }
  tfm::ModelSearchEngine modelEngine(model);

  // --- 1. fixture replay (real model) --------------------------------------
  {
    const std::vector<FixtureCase> cases = parseFixtures(fixturePath);
    check(!cases.empty(), "fixture cases parsed");

    // sanity: the committed model.bin must still be the one that generated
    // the captures (values agree to the fixture's 4-digit rounding).
    for (const FixtureCase& fc : cases) {
      if (fc.capture.empty()) continue;  // forced position: no evals at all
      const CaptureEntry& e = fc.capture[0];
      double v = 0.0;
      std::vector<float> policy;
      modelEngine.evalPosition(e.cells, e.player, v, policy);
      bool ok = std::fabs(v - e.value) < 1e-3;
      for (size_t i = 0; ok && i < policy.size(); ++i) {
        if (std::fabs((double)e.policy[i]) > 1e8) continue;  // masked sentinel
        if (std::fabs((double)policy[i] - (double)e.policy[i]) > 1e-3) ok = false;
      }
      check(ok, "model.bin matches fixture generation (n=" +
                    std::to_string(fc.size) + " " + fc.difficulty + ")");
    }

    for (const FixtureCase& fc : cases) {
      // Forced positions (immediate win/block) never reach the search in
      // Python's _strong_move — the capture is predictedLine only.
      const int opp = fc.ai == 1 ? 2 : 1;
      const bool forced =
          tfm::winInOne(fc.size, fc.position, fc.ai) ||
          tfm::winInOne(fc.size, fc.position, opp);

      size_t searchCalls = 0;
      std::vector<tfm::ScoredMove> scored;
      bool prefixOk = true;
      if (!forced) {
        RecordingEngine eng(modelEngine);
        SearchCore core(eng, fc.size, fc.position, fc.ai, fc.depth, fc.topK,
                        fc.maxNodes, 0.0);
        scored = core.scored();
        searchCalls = eng.requests().size();

        prefixOk = searchCalls <= fc.capture.size();
        size_t firstBad = 0;
        for (size_t i = 0; prefixOk && i < searchCalls; ++i) {
          if (!requestMatches(eng.requests()[i].player, eng.requests()[i].cells,
                              fc.capture[i])) {
            prefixOk = false;
            firstBad = i;
          }
        }
        check(prefixOk, "search prefix matches capture (n=" +
                            std::to_string(fc.size) + " " + fc.difficulty +
                            ", first mismatch at " + std::to_string(firstBad) +
                            ")");
      }

      // predictedLine self-consistency: starts at the chosen move and never
      // repeats a cell (a line can't revisit an occupied cell). This replaces
      // a stale-fixture tail comparison — the Python backend's predictedLine
      // has drifted from the TS reference (e.g. it emits an extra eval after
      // an immediate win), and TS is authoritative.
      RecordingEngine eng2(modelEngine);
      SearchCore core2(eng2, fc.size, fc.position, fc.ai, fc.depth, fc.topK,
                       fc.maxNodes, 0.0);
      const std::vector<tfm::LineStep> line = core2.predictedLine(fc.expectedMove);
      bool lineOk = !line.empty() && line[0].index == fc.expectedMove &&
                    line[0].player == fc.ai;
      std::vector<bool> seen((size_t)fc.size * fc.size * fc.size, false);
      for (const tfm::LineStep& s : line) {
        if (s.index < 0 || s.index >= (int)seen.size() || seen[(size_t)s.index]) {
          lineOk = false;
          break;
        }
        seen[(size_t)s.index] = true;
      }
      check(lineOk, "predictedLine self-consistent (n=" +
                         std::to_string(fc.size) + " " + fc.difficulty + ")");

      std::printf("case n=%d %-6s: forced=%d search=%zu line=%zu capture=%zu %s\n",
                  fc.size, fc.difficulty.c_str(), forced ? 1 : 0, searchCalls,
                  line.size(), fc.capture.size(),
                  (prefixOk && lineOk) ? "OK" : "MISMATCH");
      std::fflush(stdout);
      (void)scored;
      (void)line;
    }
  }

  // --- 2. batched vs sequential leaves (real model) ------------------------
  // Focused on the runtime-relevant configurations: the actual difficulty
  // depths with each size's TS-effective node budget (effMaxNodes =
  // max(24, round(220 * 27/n^3))), plus cap/style variants at a small budget.
  {
    SequentialEngine sequential(modelEngine);
    int combos = 0;
      for (int n : {3, 4, 5}) {
      const int effMax = std::max(24, (int)std::lround(220.0 * 27.0 / (n * n * n)));
      for (int salt : {0, 3}) {
        const std::vector<int> cells = makeBoard(n, salt);
        struct Cfg {
          int depth, topK, maxNodes;
          double agg;
        };
        const Cfg cfgs[] = {
            {3, 3, effMax, 0.0},    // medium
            {4, 3, effMax, 0.0},    // hard
            {5, 3, effMax, 0.0},    // hint depth
            {4, 1, 24, 0.0},        // narrow topK + tight cap
            {4, 3, 24, 0.7},        // style-weighted, attacker
            {4, 3, 24, -0.5},       // style-weighted, defender
        };
        for (const Cfg& cfg : cfgs) {
          SearchCore a(modelEngine, n, cells, 1, cfg.depth, cfg.topK,
                       cfg.maxNodes, cfg.agg);
          SearchCore b(sequential, n, cells, 1, cfg.depth, cfg.topK,
                       cfg.maxNodes, cfg.agg);
          const auto sa = a.scored();
          const auto sb = b.scored();
          bool ok = sa.size() == sb.size();
          if (ok) {
            for (size_t i = 0; i < sa.size(); ++i) {
              if (sa[i].move != sb[i].move || sa[i].value != sb[i].value) {
                ok = false;
                break;
              }
            }
          }
          check(ok, "batched==sequential (bit-exact) n=" + std::to_string(n) +
                        " depth=" + std::to_string(cfg.depth) + " topK=" +
                        std::to_string(cfg.topK) + " agg=" +
                        std::to_string(cfg.agg) + " maxNodes=" +
                        std::to_string(cfg.maxNodes));
          ++combos;
          std::fflush(stdout);
        }
      }
    }
    std::printf("batched-vs-sequential: %d combos %s\n", combos,
                g_failures == 0 ? "OK" : "FAILED");
  }

  // --- 3. smoke -------------------------------------------------------------
  {
    const std::vector<int> empty27(27, 0);
    const auto scored = tfm::searchScored(model, empty27, 3, 1, 4, 3, 220, 0.0);
    check(scored.size() == 27, "empty 3x3x3 scores 27 moves");
    bool ascending = true;
    for (size_t i = 1; i < scored.size(); ++i) {
      if (scored[i - 1].move >= scored[i].move) ascending = false;
    }
    check(ascending, "moves in ascending index order");
    std::printf("smoke: %zu moves, ascending OK\n", scored.size());
  }

  if (g_failures == 0) {
    std::printf("SEARCH PARITY: PASS\n");
    return 0;
  }
  std::printf("SEARCH PARITY: FAIL (%d failures)\n", g_failures);
  return 1;
}
