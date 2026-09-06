#include <ReactCommon/CallInvoker.h>
#include <react/nativemodule/core/ReactCommon/CxxTurboModuleUtils.h>
#include <TfmEngine.h>

namespace {

struct TfmEngineRegistrar {
  TfmEngineRegistrar() {
    facebook::react::registerCxxModuleToGlobalModuleMap(
        tfmengine::TfmEngineTurboModule::kModuleName,
        [](std::shared_ptr<facebook::react::CallInvoker> jsInvoker) {
          return std::make_shared<tfmengine::TfmEngineTurboModule>(
              std::move(jsInvoker));
        });
  }
};

TfmEngineRegistrar g_tfmEngineRegistrar;

}  // namespace