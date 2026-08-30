#pragma once

#include <jsi/jsi.h>
#include <ReactCommon/CallInvoker.h>
#include <ReactCommon/TurboModule.h>
#include <ReactCommon/TurboModuleWithJSIBindings.h>

#include <memory>

namespace tfmengine {

// Shared, mutex-guarded model holder. Definition lives in TfmEngine.cpp.
struct EngineState;

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
  static facebook::jsi::Value numelHost(
      facebook::jsi::Runtime& rt,
      facebook::react::TurboModule& module,
      const facebook::jsi::Value* args,
      size_t count);
};

}  // namespace tfmengine