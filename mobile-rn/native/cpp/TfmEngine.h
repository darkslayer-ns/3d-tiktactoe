#pragma once

#include <jsi/jsi.h>
#include <ReactCommon/CallInvoker.h>
#include <ReactCommon/TurboModule.h>
#include <ReactCommon/TurboModuleWithJSIBindings.h>

#include <memory>

namespace tfmengine {

// Shared, mutex-guarded model holder. Definition lives in TfmEngine.cpp.
struct EngineState;

/**
 * Property names exposed by the `globalThis.__TfmEngine` host object, as a
 * type-safe enum. The JSI `get` dispatch resolves a `PropNameID` to one of
 * these once, then switches on it — no repeated string comparisons.
 */
enum class TfmMethod {
  Unknown,
  Load,
  EvalPosition,
  EvalPositions,
  Numel,
  SearchScored,
  PredictedLine,
};

// Installs `globalThis.__TfmEngine`, a jsi::HostObject exposing the three host
// functions (load / evalPosition / numel) that wrap the compiled tfm::Model
// engine. Idempotent. The JS side (src/native/TfmEngine.ts) reads exactly this
// global name.
void installTfmEngine(facebook::jsi::Runtime& runtime);

// C++ TurboModule glue for the New Architecture. When JS first requests the
// "TfmEngine" module, the runtime installs the JSI bindings above (via
// TurboModuleWithJSIBindings::installJSIBindings). Also mirrors the three host
// functions so the module works when called directly through TurboModuleProxy.
//
// Android: registered from the app's copied OnLoad.cpp cxxModuleProvider.
// iOS:    registered through the ReactNativeFactory delegate (see README).
class TfmEngineTurboModule
    : public facebook::react::TurboModule,
      public facebook::react::TurboModuleWithJSIBindings {
 public:
  static constexpr char kModuleName[] = "TfmEngine";

  explicit TfmEngineTurboModule(
      std::shared_ptr<facebook::react::CallInvoker> jsInvoker);

  void installJSIBindingsWithRuntime(facebook::jsi::Runtime& runtime) override;

 private:
  std::shared_ptr<EngineState> state_;

  static facebook::jsi::Value loadHost(
      facebook::jsi::Runtime& rt,
      facebook::react::TurboModule& module,
      const facebook::jsi::Value* args,
      size_t count);
  static facebook::jsi::Value evalPositionHost(
      facebook::jsi::Runtime& rt,
      facebook::react::TurboModule& module,
      const facebook::jsi::Value* args,
      size_t count);
  static facebook::jsi::Value evalPositionsHost(
      facebook::jsi::Runtime& rt,
      facebook::react::TurboModule& module,
      const facebook::jsi::Value* args,
      size_t count);
  static facebook::jsi::Value numelHost(
      facebook::jsi::Runtime& rt,
      facebook::react::TurboModule& module,
      const facebook::jsi::Value* args,
      size_t count);
  static facebook::jsi::Value searchScoredHost(
      facebook::jsi::Runtime& rt,
      facebook::react::TurboModule& module,
      const facebook::jsi::Value* args,
      size_t count);
  /** Async variant of searchScoredHost: runs the search on a background thread
   *  and resolves a JS Promise (keeps the JS thread / UI responsive). */
  static facebook::jsi::Value searchScoredAsyncHost(
      facebook::jsi::Runtime& rt,
      facebook::react::TurboModule& module,
      const facebook::jsi::Value* args,
      size_t count);
  static facebook::jsi::Value predictedLineHost(
      facebook::jsi::Runtime& rt,
      facebook::react::TurboModule& module,
      const facebook::jsi::Value* args,
      size_t count);
};

}  // namespace tfmengine