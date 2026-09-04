#include "TfmEngine.h"

#include <jsi/jsi.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <pthread.h>
#include <sys/resource.h>

#include "tfm/model.hpp"
#include "tfm/search.hpp"
#include "tfm_memory_loader.hpp"
#include "tfm_model_data.h"

namespace tfmengine {

using namespace facebook;

// ---------------------------------------------------------------------------
// Shared engine state + host function implementations.
// ---------------------------------------------------------------------------

struct EngineState {
  std::mutex mu;
  std::mutex infer;  // serializes forward passes vs the background search
  std::shared_ptr<tfm::Model> model;
  bool loaded = false;
};

namespace {

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

jsi::Value hostLoad(jsi::Runtime& rt, EngineState& st) {
  (void)rt;  // load() reports success via the boolean return, not an error
  std::string err;
  if (!ensureLoaded(st, &err)) {
    // load() is a boolean-success contract, not an error contract.
    return jsi::Value(false);
  }
  return jsi::Value(true);
}

int64_t boardSize(int n) { return (int64_t)n * n * n; }

jsi::Value hostEvalPosition(jsi::Runtime& rt, EngineState& st,
                            const jsi::Value& boardValue,
                            const jsi::Value& maskValue,
                            const jsi::Value& nValue) {
  const int n = static_cast<int>(nValue.asNumber());
  if (n < 1 || n > 6) {
    throw jsi::JSError(rt, "TfmEngine evalPosition: n must be in [1, 6]");
  }
  const size_t N = (size_t)boardSize(n);

  jsi::Array boardArr = boardValue.asObject(rt).asArray(rt);
  jsi::Array maskArr = maskValue.asObject(rt).asArray(rt);
  if (boardArr.length(rt) != N || maskArr.length(rt) != N) {
    throw jsi::JSError(rt,
                       "TfmEngine evalPosition: board and mask must each have "
                       "length n^3");
  }

  std::vector<int> board(N);
  std::vector<uint8_t> mask(N);
  for (size_t i = 0; i < N; ++i) {
    const double b = boardArr.getValueAtIndex(rt, (int)i).asNumber();
    const double m = maskArr.getValueAtIndex(rt, (int)i).asNumber();
    if (b < 0.0 || b > 2.0 || b != std::floor(b)) {
      throw jsi::JSError(rt,
                         "TfmEngine evalPosition: board tokens must be "
                         "integers in {0, 1, 2}");
    }
    board[i] = static_cast<int>(b);
    mask[i] = m != 0.0 ? 1 : 0;
  }

  // Auto-load if the caller skipped load(). ensureLoaded is idempotent and
  // takes the mutex itself.
  std::string err;
  if (!ensureLoaded(st, &err)) {
    throw jsi::JSError(rt, "TfmEngine evalPosition: " + err);
  }
  std::lock_guard<std::mutex> inferLock(st.infer);

  // forward() is thread-safe (all temporaries are stack/local); copy the
  // shared model and run outside the lock.
  std::shared_ptr<tfm::Model> model;
  {
    std::lock_guard<std::mutex> lock(st.mu);
    model = st.model;
  }

  std::vector<float> policy(N);
  float value = 0.0f;
  model->forward(board.data(), mask.data(), n, value, policy.data());

  jsi::Object result(rt);
  result.setProperty(rt, "value", value);
  jsi::Array policyArr(rt, (int)N);
  for (size_t i = 0; i < N; ++i) {
    policyArr.setValueAtIndex(rt, (int)i, policy[i]);
  }
  result.setProperty(rt, "policy", policyArr);
  return result;
}

jsi::Value hostNumel(jsi::Runtime& rt, EngineState& st) {
  std::string err;
  if (!ensureLoaded(st, &err)) {
    throw jsi::JSError(rt, "TfmEngine numel: " + err);
  }
  std::lock_guard<std::mutex> lock(st.mu);
  return jsi::Value(st.model->numel());
}

jsi::Value hostEvalPositions(jsi::Runtime& rt, EngineState& st,
                             const jsi::Value& boardsValue,
                             const jsi::Value& maskValue,
                             const jsi::Value& nValue) {
  const int n = static_cast<int>(nValue.asNumber());
  if (n < 1 || n > 6) {
    throw jsi::JSError(rt, "TfmEngine evalPositions: n must be in [1, 6]");
  }
  const size_t N = (size_t)boardSize(n);

  jsi::Array boardsArr = boardsValue.asObject(rt).asArray(rt);
  jsi::Array maskArr = maskValue.asObject(rt).asArray(rt);
  const size_t total = (size_t)boardsArr.length(rt);
  if (total == 0 || total % N != 0 || (size_t)maskArr.length(rt) != total) {
    throw jsi::JSError(
        rt, "TfmEngine evalPositions: boards/mask lengths must be equal and "
            "multiples of n^3");
  }
  const int count = (int)(total / N);

  std::vector<int> boards(total);
  std::vector<uint8_t> masks(total);
  for (size_t i = 0; i < total; ++i) {
    const double b = boardsArr.getValueAtIndex(rt, (int)i).asNumber();
    const double m = maskArr.getValueAtIndex(rt, (int)i).asNumber();
    if (b < 0.0 || b > 2.0 || b != std::floor(b)) {
      throw jsi::JSError(rt,
                         "TfmEngine evalPositions: board tokens must be "
                         "integers in {0, 1, 2}");
    }
    boards[i] = static_cast<int>(b);
    masks[i] = m != 0.0 ? 1 : 0;
  }

  std::string err;
  if (!ensureLoaded(st, &err)) {
    throw jsi::JSError(rt, "TfmEngine evalPositions: " + err);
  }
  std::lock_guard<std::mutex> inferLock(st.infer);
  std::shared_ptr<tfm::Model> model;
  {
    std::lock_guard<std::mutex> lock(st.mu);
    model = st.model;
  }

  std::vector<float> values(count);
  std::vector<float> policies(total);
  model->forwardBatch(n, count, boards.data(), masks.data(), values.data(),
                      policies.data());

  jsi::Object result(rt);
  jsi::Array valuesArr(rt, count);
  for (int i = 0; i < count; ++i) valuesArr.setValueAtIndex(rt, i, values[i]);
  result.setProperty(rt, "values", valuesArr);
  jsi::Array policyArr(rt, (int)total);
  for (size_t i = 0; i < total; ++i) {
    policyArr.setValueAtIndex(rt, (int)i, policies[i]);
  }
  result.setProperty(rt, "policies", policyArr);
  return result;
}

jsi::Value hostSearchScored(jsi::Runtime& rt, std::shared_ptr<EngineState> state,
                            std::shared_ptr<react::CallInvoker> jsInvoker,
                            const jsi::Value& cellsValue,
                            const jsi::Value& aiValue,
                            const jsi::Value& depthValue,
                            const jsi::Value& topKValue,
                            const jsi::Value& maxNodesValue,
                            const jsi::Value& aggValue,
                            const jsi::Value& nValue) {
  const int n = static_cast<int>(nValue.asNumber());
  if (n < 1 || n > 6) {
    throw jsi::JSError(rt, "TfmEngine searchScored: n must be in [1, 6]");
  }
  const size_t N = (size_t)n * n * n;
  jsi::Array cellsArr = cellsValue.asObject(rt).asArray(rt);
  if (cellsArr.length(rt) != N) {
    throw jsi::JSError(rt, "TfmEngine searchScored: cells must have length n^3");
  }
  std::vector<int> cells(N);
  for (size_t i = 0; i < N; ++i) {
    const double c = cellsArr.getValueAtIndex(rt, (int)i).asNumber();
    if (c < 0.0 || c > 2.0 || c != std::floor(c)) {
      throw jsi::JSError(rt, "TfmEngine searchScored: cells must be {0,1,2}");
    }
    cells[i] = static_cast<int>(c);
  }

  std::string err;
  if (!ensureLoaded(*state, &err)) {
    throw jsi::JSError(rt, "TfmEngine searchScored: " + err);
  }
  std::shared_ptr<tfm::Model> model;
  {
    std::lock_guard<std::mutex> lock(state->mu);
    model = state->model;
  }

  const int ai = static_cast<int>(aiValue.asNumber());
  const int depth = static_cast<int>(depthValue.asNumber());
  const int topK = static_cast<int>(topKValue.asNumber());
  const int maxNodes = static_cast<int>(maxNodesValue.asNumber());
  const double aggression = aggValue.asNumber();

  auto buildResult = [&rt](const std::vector<tfm::ScoredMove>& scored) {
    jsi::Object result(rt);
    jsi::Array movesArr(rt, (int)scored.size());
    jsi::Array valuesArr(rt, (int)scored.size());
    for (size_t i = 0; i < scored.size(); ++i) {
      movesArr.setValueAtIndex(rt, (int)i, scored[i].move);
      valuesArr.setValueAtIndex(rt, (int)i, scored[i].value);
    }
    result.setProperty(rt, "moves", movesArr);
    result.setProperty(rt, "values", valuesArr);
    return result;
  };

  // Plain-JSI install (no CallInvoker): compute synchronously — the caller's
  // `await` still works on the plain object.
  if (!jsInvoker) {
    std::lock_guard<std::mutex> inferLock(state->infer);
    return buildResult(tfm::searchScored(*model, cells, n, ai, depth, topK,
                                         maxNodes, aggression));
  }

  // Run the whole lookahead OFF the JS thread and resolve a promise on it, so
  // a slow 4x4x4/5x5x5 search never blocks the click/render path.
  jsi::Function promiseCtor = rt.global().getPropertyAsFunction(rt, "Promise");
  jsi::Function executor = jsi::Function::createFromHostFunction(
      rt, jsi::PropNameID::forAscii(rt, "executor"), 2,
      [state, jsInvoker, model, cells = std::move(cells), n, ai, depth, topK,
       maxNodes, aggression](jsi::Runtime& runtime, const jsi::Value&,
                             const jsi::Value* execArgs,
                             size_t) -> jsi::Value {
        auto resolve = std::make_shared<jsi::Function>(
            execArgs[0].getObject(runtime).asFunction(runtime));
        auto reject = std::make_shared<jsi::Function>(
            execArgs[1].getObject(runtime).asFunction(runtime));
        jsi::Runtime* rtPtr = &runtime;
        std::thread([state, jsInvoker, model, resolve, reject, rtPtr, cells, n,
                     ai, depth, topK, maxNodes, aggression]() {
          // Background search must NOT starve the UI/main thread: run at a
          // lower scheduling priority so clicks/render stay smooth.
#if defined(__APPLE__)
          pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#else
          setpriority(PRIO_PROCESS, 0, 10);
#endif
          try {
            std::vector<tfm::ScoredMove> scored;
            {
              std::lock_guard<std::mutex> inferLock(state->infer);
              scored = tfm::searchScored(*model, cells, n, ai, depth, topK,
                                         maxNodes, aggression);
            }
            jsInvoker->invokeAsync(
                [rtPtr, resolve, reject, scored = std::move(scored)]() {
                  jsi::Runtime& rt2 = *rtPtr;
                  try {
                    jsi::Object result(rt2);
                    jsi::Array movesArr(rt2, (int)scored.size());
                    jsi::Array valuesArr(rt2, (int)scored.size());
                    for (size_t i = 0; i < scored.size(); ++i) {
                      movesArr.setValueAtIndex(rt2, (int)i, scored[i].move);
                      valuesArr.setValueAtIndex(rt2, (int)i, scored[i].value);
                    }
                    result.setProperty(rt2, "moves", movesArr);
                    result.setProperty(rt2, "values", valuesArr);
                    resolve->call(rt2, result);
                  } catch (const std::exception& e) {
                    try {
                      reject->call(rt2,
                                   jsi::String::createFromUtf8(rt2, e.what()));
                    } catch (...) {
                    }
                  }
                });
          } catch (const std::exception& e) {
            std::string msg = e.what();
            jsInvoker->invokeAsync([rtPtr, reject, msg]() {
              try {
                reject->call(*rtPtr, jsi::String::createFromUtf8(*rtPtr, msg));
              } catch (...) {
              }
            });
          }
        }).detach();
        return jsi::Value::undefined();
      });
  return promiseCtor.callAsConstructor(rt, executor);
}

jsi::Value hostPredictedLine(jsi::Runtime& rt, EngineState& st,
                             const jsi::Value& cellsValue,
                             const jsi::Value& aiValue,
                             const jsi::Value& chosenValue,
                             const jsi::Value& depthValue,
                             const jsi::Value& nValue) {
  const int n = static_cast<int>(nValue.asNumber());
  if (n < 1 || n > 6) {
    throw jsi::JSError(rt, "TfmEngine predictedLine: n must be in [1, 6]");
  }
  const size_t N = (size_t)n * n * n;
  jsi::Array cellsArr = cellsValue.asObject(rt).asArray(rt);
  if (cellsArr.length(rt) != N) {
    throw jsi::JSError(rt, "TfmEngine predictedLine: cells must have length n^3");
  }
  std::vector<int> cells(N);
  for (size_t i = 0; i < N; ++i) {
    const double c = cellsArr.getValueAtIndex(rt, (int)i).asNumber();
    if (c < 0.0 || c > 2.0 || c != std::floor(c)) {
      throw jsi::JSError(rt, "TfmEngine predictedLine: cells must be {0,1,2}");
    }
    cells[i] = static_cast<int>(c);
  }

  std::string err;
  if (!ensureLoaded(st, &err)) {
    throw jsi::JSError(rt, "TfmEngine predictedLine: " + err);
  }
  std::lock_guard<std::mutex> inferLock(st.infer);
  std::shared_ptr<tfm::Model> model;
  {
    std::lock_guard<std::mutex> lock(st.mu);
    model = st.model;
  }

  const int ai = static_cast<int>(aiValue.asNumber());
  const int chosen = static_cast<int>(chosenValue.asNumber());
  const int depth = static_cast<int>(depthValue.asNumber());

  const std::vector<tfm::LineStep> line =
      tfm::predictedLine(*model, cells, n, ai, chosen, depth);

  jsi::Object result(rt);
  jsi::Array playersArr(rt, (int)line.size());
  jsi::Array indicesArr(rt, (int)line.size());
  for (size_t i = 0; i < line.size(); ++i) {
    playersArr.setValueAtIndex(rt, (int)i, line[i].player);
    indicesArr.setValueAtIndex(rt, (int)i, line[i].index);
  }
  result.setProperty(rt, "players", playersArr);
  result.setProperty(rt, "indices", indicesArr);
  return result;
}

jsi::Value makeHostFunction(
    jsi::Runtime& rt, const jsi::PropNameID& name, unsigned int argCount,
    std::function<jsi::Value(jsi::Runtime&, const jsi::Value*, size_t)> fn) {
  return jsi::Function::createFromHostFunction(
      rt, name, argCount,
      [fn = std::move(fn)](jsi::Runtime& runtime, const jsi::Value&,
                           const jsi::Value* args, size_t count) -> jsi::Value {
        return fn(runtime, args, count);
      });
}

}  // namespace

// ---------------------------------------------------------------------------
// HostObject installed as globalThis.__TfmEngine.
// ---------------------------------------------------------------------------

namespace {

// Resolve a JS property name to its type-safe method enum exactly once. The
// enum + its JS name live side-by-side in ONE table, and the host-object
// dispatch below switches on the result. (C++ cannot switch on a string or a
// jsi::PropNameID, so this name→enum lookup is the single unavoidable point.)
struct MethodEntry {
  TfmMethod method;
  const char* name;
};

constexpr MethodEntry kMethodTable[] = {
    {TfmMethod::Load, "load"},
    {TfmMethod::EvalPosition, "evalPosition"},
    {TfmMethod::EvalPositions, "evalPositions"},
    {TfmMethod::Numel, "numel"},
    {TfmMethod::SearchScored, "searchScored"},
    {TfmMethod::PredictedLine, "predictedLine"},
};
constexpr size_t kMethodCount = sizeof(kMethodTable) / sizeof(kMethodTable[0]);

TfmMethod tfmMethodFromName(const std::string& n) {
  for (const auto& e : kMethodTable) {
    if (n == e.name) return e.method;
  }
  return TfmMethod::Unknown;
}

}  // namespace

class TfmEngineHostObject : public jsi::HostObject {
 public:
  explicit TfmEngineHostObject(std::shared_ptr<react::CallInvoker> jsInvoker)
      : state_(std::make_shared<EngineState>()),
        jsInvoker_(std::move(jsInvoker)) {}

  jsi::Value get(jsi::Runtime& rt, const jsi::PropNameID& name) override {
    // Capture the shared_ptrs (not references) so a function returned to JS
    // keeps the engine + invoker alive even if the global is dropped.
    std::shared_ptr<EngineState> state = state_;
    std::shared_ptr<react::CallInvoker> jsInvoker = jsInvoker_;
    switch (tfmMethodFromName(name.utf8(rt))) {
      case TfmMethod::Load:
        return makeHostFunction(rt, name, 0, [state](jsi::Runtime& r,
                                                     const jsi::Value*,
                                                     size_t) -> jsi::Value {
          return hostLoad(r, *state);
        });
      case TfmMethod::EvalPosition:
        return makeHostFunction(rt, name, 3, [state](jsi::Runtime& r,
                                                     const jsi::Value* args,
                                                     size_t) -> jsi::Value {
          return hostEvalPosition(r, *state, args[0], args[1], args[2]);
        });
      case TfmMethod::EvalPositions:
        return makeHostFunction(rt, name, 3, [state](jsi::Runtime& r,
                                                     const jsi::Value* args,
                                                     size_t) -> jsi::Value {
          return hostEvalPositions(r, *state, args[0], args[1], args[2]);
        });
      case TfmMethod::Numel:
        return makeHostFunction(rt, name, 0, [state](jsi::Runtime& r,
                                                     const jsi::Value*,
                                                     size_t) -> jsi::Value {
          return hostNumel(r, *state);
        });
      case TfmMethod::SearchScored:
        return makeHostFunction(rt, name, 7,
                                [state, jsInvoker](jsi::Runtime& r,
                                                   const jsi::Value* args,
                                                   size_t) -> jsi::Value {
                                  return hostSearchScored(
                                      r, state, jsInvoker, args[0], args[1],
                                      args[2], args[3], args[4], args[5],
                                      args[6]);
                                });
      case TfmMethod::PredictedLine:
        return makeHostFunction(rt, name, 5, [state](jsi::Runtime& r,
                                                     const jsi::Value* args,
                                                     size_t) -> jsi::Value {
          return hostPredictedLine(r, *state, args[0], args[1], args[2],
                                   args[3], args[4]);
        });
      case TfmMethod::Unknown:
      default:
        return jsi::Value::undefined();
    }
  }

  std::vector<jsi::PropNameID> getPropertyNames(jsi::Runtime& rt) override {
    std::vector<jsi::PropNameID> names;
    names.reserve(kMethodCount);
    for (const auto& e : kMethodTable) {
      names.push_back(jsi::PropNameID::forUtf8(rt, e.name));
    }
    return names;
  }

 private:
  std::shared_ptr<EngineState> state_;
  std::shared_ptr<react::CallInvoker> jsInvoker_;
};

void installTfmEngine(jsi::Runtime& runtime,
                      std::shared_ptr<react::CallInvoker> jsInvoker) {
  if (runtime.global().hasProperty(runtime, "__TfmEngine")) {
    return;
  }
  auto host =
      std::make_shared<TfmEngineHostObject>(std::move(jsInvoker));
  auto object = jsi::Object::createFromHostObject(runtime, host);
  runtime.global().setProperty(runtime, "__TfmEngine", std::move(object));
}

// ---------------------------------------------------------------------------
// TurboModule glue (New Architecture).
// ---------------------------------------------------------------------------

TfmEngineTurboModule::TfmEngineTurboModule(
    std::shared_ptr<react::CallInvoker> jsInvoker)
    : react::TurboModule(kModuleName, std::move(jsInvoker)),
      state_(std::make_shared<EngineState>()) {
  methodMap_["load"] =
      react::TurboModule::MethodMetadata{0, &TfmEngineTurboModule::loadHost};
  methodMap_["evalPosition"] = react::TurboModule::MethodMetadata{
      3, &TfmEngineTurboModule::evalPositionHost};
  methodMap_["evalPositions"] = react::TurboModule::MethodMetadata{
      3, &TfmEngineTurboModule::evalPositionsHost};
  methodMap_["numel"] =
      react::TurboModule::MethodMetadata{0, &TfmEngineTurboModule::numelHost};
  methodMap_["searchScored"] = react::TurboModule::MethodMetadata{
      7, &TfmEngineTurboModule::searchScoredHost};
  methodMap_["predictedLine"] = react::TurboModule::MethodMetadata{
      5, &TfmEngineTurboModule::predictedLineHost};
}

void TfmEngineTurboModule::installJSIBindingsWithRuntime(
    jsi::Runtime& runtime) {
  installTfmEngine(runtime, jsInvoker_);
}

jsi::Value TfmEngineTurboModule::loadHost(
    jsi::Runtime& rt, react::TurboModule& module, const jsi::Value*, size_t) {
  auto& self = static_cast<TfmEngineTurboModule&>(module);
  return hostLoad(rt, *self.state_);
}

jsi::Value TfmEngineTurboModule::evalPositionHost(
    facebook::jsi::Runtime& rt, react::TurboModule& module, const jsi::Value* args,
    size_t) {
  auto& self = static_cast<TfmEngineTurboModule&>(module);
  return hostEvalPosition(rt, *self.state_, args[0], args[1], args[2]);
}

jsi::Value TfmEngineTurboModule::evalPositionsHost(
    facebook::jsi::Runtime& rt, react::TurboModule& module, const jsi::Value* args,
    size_t) {
  auto& self = static_cast<TfmEngineTurboModule&>(module);
  return hostEvalPositions(rt, *self.state_, args[0], args[1], args[2]);
}

jsi::Value TfmEngineTurboModule::numelHost(
    jsi::Runtime& rt, react::TurboModule& module, const jsi::Value*, size_t) {
  auto& self = static_cast<TfmEngineTurboModule&>(module);
  return hostNumel(rt, *self.state_);
}

jsi::Value TfmEngineTurboModule::searchScoredHost(
    jsi::Runtime& rt, react::TurboModule& module, const jsi::Value* args,
    size_t) {
  auto& self = static_cast<TfmEngineTurboModule&>(module);
  return hostSearchScored(rt, self.state_, self.jsInvoker_, args[0], args[1],
                          args[2], args[3], args[4], args[5], args[6]);
}

jsi::Value TfmEngineTurboModule::predictedLineHost(
    jsi::Runtime& rt, react::TurboModule& module, const jsi::Value* args,
    size_t) {
  auto& self = static_cast<TfmEngineTurboModule&>(module);
  return hostPredictedLine(rt, *self.state_, args[0], args[1], args[2], args[3],
                           args[4]);
}

}  // namespace tfmengine