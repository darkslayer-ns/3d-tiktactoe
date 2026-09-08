#pragma once

#include <jsi/jsi.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "NativeAI.h"

namespace tfmengine {

// Shared, mutex-guarded model + AI-session holder. Owned by both the JSI host
// object and the C++ TurboModule (they share one EngineState instance).
struct EngineState {
  std::atomic_bool alive{true};
  std::mutex mu;     // guards `model` / `loaded`
  std::mutex infer;  // serializes forward passes vs the background search
  std::shared_ptr<tfm::Model> model;
  std::shared_ptr<NativeAI> ai;
  std::mutex aiMu;  // guards `ai`
  bool loaded = false;
};

// Parses the embedded weights into the model exactly once (idempotent). No
// filesystem I/O: the weights are an embedded C array, parsed in memory.
bool ensureLoaded(EngineState& st, std::string* err);

// ---- JSI <-> C++ conversion helpers shared by the host functions -----------

std::vector<double> readNumberArray(facebook::jsi::Runtime& rt,
                                    const facebook::jsi::Value& value);
double objectNumber(facebook::jsi::Runtime& rt, const facebook::jsi::Object& obj,
                    const char* name, double fallback);
std::string objectString(facebook::jsi::Runtime& rt, const facebook::jsi::Object& obj,
                         const char* name, const char* fallback);
NativeAIState readAIState(facebook::jsi::Runtime& rt,
                          const facebook::jsi::Object& obj);
facebook::jsi::Array numberArray(facebook::jsi::Runtime& rt,
                                 const std::vector<double>& values);
facebook::jsi::Value aiStateValue(facebook::jsi::Runtime& rt,
                                  const NativeAIState& state);
facebook::jsi::Value aiKnowledgeValue(facebook::jsi::Runtime& rt,
                                      const NativeAIKnowledge& knowledge);
facebook::jsi::Value aiResultValue(facebook::jsi::Runtime& rt,
                                   const NativeAIResult& result);

}  // namespace tfmengine
