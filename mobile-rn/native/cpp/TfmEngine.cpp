#include "TfmEngine.h"

#include <jsi/jsi.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "tfm/model.hpp"
#include "tfm_memory_loader.hpp"
#include "tfm_model_data.h"

namespace tfmengine {

using namespace facebook;

// ---------------------------------------------------------------------------
// Shared engine state + host function implementations.
// ---------------------------------------------------------------------------

struct EngineState {
  std::mutex mu;
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

class TfmEngineHostObject : public jsi::HostObject {
 public:
  TfmEngineHostObject() : state_(std::make_shared<EngineState>()) {}

  jsi::Value get(jsi::Runtime& rt, const jsi::PropNameID& name) override {
    const std::string n = name.utf8(rt);
    // Capture the shared_ptr (not a reference) so a function returned to JS
    // keeps the engine alive even if the __TfmEngine global is dropped.
    std::shared_ptr<EngineState> state = state_;
    if (n == "load") {
      return makeHostFunction(rt, name, 0, [state](jsi::Runtime& r,
                                                   const jsi::Value*,
                                                   size_t) -> jsi::Value {
        return hostLoad(r, *state);
      });
    }
    if (n == "evalPosition") {
      return makeHostFunction(rt, name, 3, [state](jsi::Runtime& r,
                                                   const jsi::Value* args,
                                                   size_t) -> jsi::Value {
        return hostEvalPosition(r, *state, args[0], args[1], args[2]);
      });
    }
    if (n == "numel") {
      return makeHostFunction(rt, name, 0, [state](jsi::Runtime& r,
                                                   const jsi::Value*,
                                                   size_t) -> jsi::Value {
        return hostNumel(r, *state);
      });
    }
    return jsi::Value::undefined();
  }

  std::vector<jsi::PropNameID> getPropertyNames(jsi::Runtime& rt) override {
    std::vector<jsi::PropNameID> names;
    names.push_back(jsi::PropNameID::forUtf8(rt, "load"));
    names.push_back(jsi::PropNameID::forUtf8(rt, "evalPosition"));
    names.push_back(jsi::PropNameID::forUtf8(rt, "numel"));
    return names;
  }

 private:
  std::shared_ptr<EngineState> state_;
};

void installTfmEngine(jsi::Runtime& runtime) {
  if (runtime.global().hasProperty(runtime, "__TfmEngine")) {
    return;
  }
  auto host = std::make_shared<TfmEngineHostObject>();
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
  methodMap_["numel"] =
      react::TurboModule::MethodMetadata{0, &TfmEngineTurboModule::numelHost};
}

void TfmEngineTurboModule::installJSIBindingsWithRuntime(
    jsi::Runtime& runtime) {
  installTfmEngine(runtime);
}

jsi::Value TfmEngineTurboModule::loadHost(
    jsi::Runtime& rt, react::TurboModule& module, const jsi::Value*, size_t) {
  auto& self = static_cast<TfmEngineTurboModule&>(module);
  return hostLoad(rt, *self.state_);
}

jsi::Value TfmEngineTurboModule::evalPositionHost(
    jsi::Runtime& rt, react::TurboModule& module, const jsi::Value* args,
    size_t) {
  auto& self = static_cast<TfmEngineTurboModule&>(module);
  return hostEvalPosition(rt, *self.state_, args[0], args[1], args[2]);
}

jsi::Value TfmEngineTurboModule::numelHost(
    jsi::Runtime& rt, react::TurboModule& module, const jsi::Value*, size_t) {
  auto& self = static_cast<TfmEngineTurboModule&>(module);
  return hostNumel(rt, *self.state_);
}

}  // namespace tfmengine