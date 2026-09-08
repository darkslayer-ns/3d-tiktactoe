#include "TfmEngineState.h"

#include "tfm_memory_loader.hpp"
#include "tfm_model_data.h"

namespace tfmengine {
using namespace facebook;

// Parses the embedded weights into the model exactly once. No filesystem I/O:
// tfm::loadWeights() is path-based (fopen), which is why we use the in-memory
// parser that understands the identical TFM1 format (see tfm_memory_loader.hpp).
bool ensureLoaded(EngineState& st, std::string* err) {
  std::lock_guard<std::mutex> lock(st.mu);
  if (st.loaded) return true;
  if (!st.model) st.model = std::make_shared<tfm::Model>();

  const std::string weights(reinterpret_cast<const char*>(tfm::kModelBin),
                            tfm::kModelBinSize);
  if (!tfm::loadWeightsFromMemory(
          reinterpret_cast<const unsigned char*>(weights.data()),
          weights.size(), *st.model, err)) {
    st.model.reset();
    return false;
  }
  st.loaded = true;
  return true;
}

std::vector<double> readNumberArray(jsi::Runtime& rt, const jsi::Value& value) {
  std::vector<double> out;
  if (!value.isObject()) return out;
  const auto arr = value.asObject(rt).asArray(rt);
  out.reserve(arr.length(rt));
  for (size_t i = 0; i < arr.length(rt); ++i) {
    out.push_back(arr.getValueAtIndex(rt, static_cast<int>(i)).asNumber());
  }
  return out;
}

double objectNumber(jsi::Runtime& rt, const jsi::Object& object,
                    const char* name, double fallback) {
  const auto value = object.getProperty(rt, name);
  return value.isNumber() ? value.asNumber() : fallback;
}

std::string objectString(jsi::Runtime& rt, const jsi::Object& object,
                         const char* name, const char* fallback) {
  const auto value = object.getProperty(rt, name);
  return value.isString() ? value.asString(rt).utf8(rt) : fallback;
}

NativeAIState readAIState(jsi::Runtime& rt, const jsi::Object& object) {
  NativeAIState state;
  state.affinity = readNumberArray(rt, object.getProperty(rt, "affinity"));
  state.profile = readNumberArray(rt, object.getProperty(rt, "profile"));
  state.perception = readNumberArray(rt, object.getProperty(rt, "perception"));
  state.stats = readNumberArray(rt, object.getProperty(rt, "stats"));
  state.adaptive = objectNumber(rt, object, "adaptive", 0.0);
  state.aggression = objectNumber(rt, object, "aggression", 0.0);
  return state;
}

jsi::Array numberArray(jsi::Runtime& rt, const std::vector<double>& values) {
  jsi::Array out(rt, static_cast<int>(values.size()));
  for (size_t i = 0; i < values.size(); ++i) out.setValueAtIndex(rt, static_cast<int>(i), values[i]);
  return out;
}

jsi::Value aiStateValue(jsi::Runtime& rt, const NativeAIState& state) {
  jsi::Object out(rt);
  out.setProperty(rt, "affinity", numberArray(rt, state.affinity));
  out.setProperty(rt, "profile", numberArray(rt, state.profile));
  out.setProperty(rt, "perception", numberArray(rt, state.perception));
  out.setProperty(rt, "stats", numberArray(rt, state.stats));
  out.setProperty(rt, "adaptive", state.adaptive);
  out.setProperty(rt, "aggression", state.aggression);
  return out;
}

jsi::Value aiKnowledgeValue(jsi::Runtime& rt, const NativeAIKnowledge& knowledge) {
  jsi::Value value = aiStateValue(rt, knowledge);
  auto out = value.asObject(rt);
  out.setProperty(rt, "winProbHuman", knowledge.winProbHuman);
  out.setProperty(rt, "winProbAi", knowledge.winProbAi);
  out.setProperty(rt, "bestMoveIndex", knowledge.bestMoveIndex);
  jsi::Array predictions(rt, static_cast<int>(knowledge.predictions.size()));
  for (size_t i = 0; i < knowledge.predictions.size(); ++i) {
    jsi::Object row(rt);
    row.setProperty(rt, "index", knowledge.predictions[i].index);
    row.setProperty(rt, "prob", knowledge.predictions[i].probability);
    predictions.setValueAtIndex(rt, static_cast<int>(i), row);
  }
  out.setProperty(rt, "predictions", predictions);
  return out;
}

jsi::Value aiResultValue(jsi::Runtime& rt, const NativeAIResult& result) {
  jsi::Object out(rt);
  out.setProperty(rt, "move", result.move);
  out.setProperty(rt, "kind", result.kind);
  out.setProperty(rt, "value", result.value);
  out.setProperty(rt, "depth", result.depth);
  jsi::Array scored(rt, static_cast<int>(result.scored.size()));
  for (size_t i = 0; i < result.scored.size(); ++i) {
    jsi::Object row(rt);
    row.setProperty(rt, "index", result.scored[i].move);
    row.setProperty(rt, "value", result.scored[i].value);
    scored.setValueAtIndex(rt, static_cast<int>(i), row);
  }
  out.setProperty(rt, "scored", scored);
  jsi::Array players(rt, static_cast<int>(result.line.size()));
  jsi::Array indices(rt, static_cast<int>(result.line.size()));
  for (size_t i = 0; i < result.line.size(); ++i) {
    players.setValueAtIndex(rt, static_cast<int>(i), result.line[i].player);
    indices.setValueAtIndex(rt, static_cast<int>(i), result.line[i].index);
  }
  out.setProperty(rt, "players", players);
  out.setProperty(rt, "indices", indices);
  return out;
}

}  // namespace tfmengine
