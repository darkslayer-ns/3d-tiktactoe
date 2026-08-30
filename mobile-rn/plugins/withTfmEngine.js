/**
 * Expo config plugin: `withTfmEngine`
 *
 * Wires the native JSI host module (mobile-rn/native) into `expo prebuild`
 * output for both platforms.
 *
 *   Android — the app uses the ReactApp Gradle plugin, which only sets an
 *   `externalNativeBuild` path if the user hasn't. We therefore:
 *     1. write `android/app/src/main/jni/CMakeLists.txt` + `OnLoad.cpp` (the
 *        documented "customize the default app CMake" pattern from
 *        ReactAndroid/cmake-utils/default-app-setup/CMakeLists.txt), where the
 *        CMake adds our `native/CMakeLists.txt` (target `tfm_engine`) and links
 *        it into `libappmodules`, and OnLoad.cpp registers the
 *        `TfmEngineTurboModule` in its `cxxModuleProvider`;
 *     2. patch `android/app/build.gradle` to point `externalNativeBuild` at
 *        `src/main/jni/CMakeLists.txt`;
 *     3. add `REACT_NATIVE_VERSION` / `REACT_NATIVE_SO_NAMES` to
 *        `gradle.properties` (best effort, harmless).
 *   If anything here doesn't match your RN/Expo version, the exact manual steps
 *   are documented in native/README.md ("Android").
 *
 *   iOS — injects `pod 'TfmEngine', :path => '../native'` into `ios/Podfile`
 *   (the podspec compiles the engine + JSI glue). Runtime registration of the
 *   TurboModule in the AppDelegate is a documented manual step (see README).
 *
 * The plugin never touches app.json.
 */

const fs = require('fs');
const path = require('path');
const {
  withDangerousMod,
  withAppBuildGradle,
  withGradleProperties,
  WarningAggregator,
} = require('@expo/config-plugins');

const ANDROID_JNI_DIR = 'app/src/main/jni';
const GRADLE_EXT_NATIVE_BUILD = `    externalNativeBuild {
        cmake {
            path "src/main/jni/CMakeLists.txt"
        }
    }`;

// ---------------------------------------------------------------------------
// Android
// ---------------------------------------------------------------------------

// The app's own CMakeLists.txt. It reproduces the RN default
// (default-app-setup/CMakeLists.txt) and additionally compiles + links
// `tfm_engine` into libappmodules. PROJECT_ROOT_DIR is <mobile-rn>/android
// (injected by the RN gradle plugin), so `${PROJECT_ROOT_DIR}/../native`
// resolves to <mobile-rn>/native.
const APP_CMAKE = `cmake_minimum_required(VERSION 3.13)

# Define the library name here.
project(appmodules)

# This file includes all the necessary to let you build your application with
# the New Architecture (copied from the RN default-app-setup CMakeLists.txt).
include(\${REACT_ANDROID_DIR}/cmake-utils/ReactNative-application.cmake)

# --- TfmEngine JSI host module (added by plugins/withTfmEngine.js) ----------
# Compiles the shared transformer engine (cpp/src/*.cpp) plus the JSI glue
# (native/cpp/TfmEngine.cpp) as a static lib and links it into libappmodules.
# The PUBLIC include dirs of tfm_engine (cpp/include, native/include,
# native/cpp) propagate here, so the OnLoad.cpp below can #include <TfmEngine.h>.
add_subdirectory(\${PROJECT_ROOT_DIR}/../native native_tfm_engine)
target_link_libraries(appmodules tfm_engine)
`;

// App-level OnLoad.cpp: identical to ReactAndroid/cmake-utils/default-app-setup/
// OnLoad.cpp but registers the TfmEngineTurboModule in cxxModuleProvider. This
// makes `global.__turboModuleProxy('TfmEngine')` return our module, which in
// turn installs `globalThis.__TfmEngine` (see installJSIBindingsWithRuntime).
const APP_ONLOAD_CPP = `// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Copied from ReactNative's default-app-setup/OnLoad.cpp and extended by
// plugins/withTfmEngine.js to register the TfmEngine JSI host module.

#include <DefaultComponentsRegistry.h>
#include <DefaultTurboModuleManagerDelegate.h>
#include <FBReactNativeSpec.h>
#include <autolinking.h>
#include <fbjni/fbjni.h>
#include <react/renderer/componentregistry/ComponentDescriptorProviderRegistry.h>

#include <TfmEngine.h>

#ifdef REACT_NATIVE_APP_CODEGEN_HEADER
#include REACT_NATIVE_APP_CODEGEN_HEADER
#endif
#ifdef REACT_NATIVE_APP_COMPONENT_DESCRIPTORS_HEADER
#include REACT_NATIVE_APP_COMPONENT_DESCRIPTORS_HEADER
#endif

namespace facebook::react {

void registerComponents(
    std::shared_ptr<const ComponentDescriptorProviderRegistry> registry) {
  // Custom Fabric Components go here.

  // We link app local components if available
#ifdef REACT_NATIVE_APP_COMPONENT_REGISTRATION
  REACT_NATIVE_APP_COMPONENT_REGISTRATION(registry);
#endif

  // And we fallback to the components autolinked
  autolinking_registerProviders(registry);
}

std::shared_ptr<TurboModule> cxxModuleProvider(
    const std::string& name,
    const std::shared_ptr<CallInvoker>& jsInvoker) {
  // TfmEngine JSI host module (native/cpp/TfmEngine.cpp)
  if (name == tfmengine::TfmEngineTurboModule::kModuleName) {
    return std::make_shared<tfmengine::TfmEngineTurboModule>(jsInvoker);
  }

  // And we fallback to the CXX module providers autolinked
  return autolinking_cxxModuleProvider(name, jsInvoker);
}

std::shared_ptr<TurboModule> javaModuleProvider(
    const std::string& name,
    const JavaTurboModule::InitParams& params) {
  // We link app local modules if available
#ifdef REACT_NATIVE_APP_MODULE_PROVIDER
  auto module = REACT_NATIVE_APP_MODULE_PROVIDER(name, params);
  if (module != nullptr) {
    return module;
  }
#endif

  // We first try to look up core modules
  if (auto module = FBReactNativeSpec_ModuleProvider(name, params)) {
    return module;
  }

  // And we fallback to the module providers autolinked
  if (auto module = autolinking_ModuleProvider(name, params)) {
    return module;
  }

  return nullptr;
}

} // namespace facebook::react

JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM* vm, void*) {
  return facebook::jni::initialize(vm, [] {
    facebook::react::DefaultTurboModuleManagerDelegate::cxxModuleProvider =
        &facebook::react::cxxModuleProvider;
    facebook::react::DefaultTurboModuleManagerDelegate::javaModuleProvider =
        &facebook::react::javaModuleProvider;
    facebook::react::DefaultComponentsRegistry::
        registerComponentDescriptorsFromEntryPoint =
            &facebook::react::registerComponents;
  });
}
`;

function writeIfAbsent(file, contents) {
  if (fs.existsSync(file)) return;
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, contents);
}

function patchAppJni(config) {
  // Dangerous mods carry their platform root on modRequest (modResults is a
  // no-op object provided by the dangerous base provider).
  const androidRoot = config.modRequest.platformProjectRoot; // <mobile-rn>/android
  const jniDir = path.join(androidRoot, ANDROID_JNI_DIR);
  writeIfAbsent(path.join(jniDir, 'CMakeLists.txt'), APP_CMAKE);
  writeIfAbsent(path.join(jniDir, 'OnLoad.cpp'), APP_ONLOAD_CPP);
  return config;
}

function patchBuildGradle(contents) {
  if (contents.includes('externalNativeBuild')) return contents; // idempotent
  const needle = 'android {';
  const idx = contents.indexOf(needle);
  if (idx === -1) return contents; // unexpected layout; README covers the manual step
  const after = contents.slice(idx + needle.length);
  return contents.slice(0, idx + needle.length) + '\n' + GRADLE_EXT_NATIVE_BUILD + after;
}

function patchGradleProperties(properties) {
  const has = (key) => properties.some((p) => p.key === key);
  if (!has('REACT_NATIVE_VERSION')) {
    properties.push({ type: 'property', key: 'REACT_NATIVE_VERSION', value: '0.86.3' });
  }
  if (!has('REACT_NATIVE_SO_NAMES')) {
    properties.push({
      type: 'property',
      key: 'REACT_NATIVE_SO_NAMES',
      value: 'libappmodules.so,libreactnative.so,libjsi.so,libhermes.so',
    });
  }
  return properties;
}

// ---------------------------------------------------------------------------
// iOS
// ---------------------------------------------------------------------------

function patchPodfile(contents) {
  if (contents.includes("pod 'TfmEngine'")) return contents; // idempotent
  const m = contents.match(/target\s+'([^']+)'\s+do/);
  if (!m) return contents;
  const needle = m[0];
  const idx = contents.indexOf(needle);
  const after = contents.slice(idx + needle.length);
  const injection = `\n  # TfmEngine JSI host module (native/TfmEngine.podspec) — added by plugins/withTfmEngine.js\n  pod 'TfmEngine', :path => '../native'\n`;
  return contents.slice(0, idx + needle.length) + injection + after;
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

module.exports = function withTfmEngine(config) {
  // Android: write app CMake + OnLoad, point externalNativeBuild at it, and
  // set RN-related gradle.properties.
  config = withDangerousMod(config, ['android', patchAppJni]);
  config = withAppBuildGradle(config, (cfg) => {
    cfg.modResults.contents = patchBuildGradle(cfg.modResults.contents);
    return cfg;
  });
  config = withGradleProperties(config, (cfg) => {
    cfg.modResults = patchGradleProperties(cfg.modResults);
    return cfg;
  });

  // iOS: make CocoaPods compile the podspec into the app.
  config = withDangerousMod(config, [
    'ios',
    async (cfg) => {
      const podfile = path.join(cfg.modRequest.platformProjectRoot, 'Podfile');
      if (!fs.existsSync(podfile)) {
        WarningAggregator.addWarningAndroid(
          'withTfmEngine',
          'ios/Podfile not found; add `pod \'TfmEngine\', :path => \'../native\'` manually (see native/README.md).',
        );
        return cfg;
      }
      const contents = fs.readFileSync(podfile, 'utf8');
      const patched = patchPodfile(contents);
      if (patched !== contents) fs.writeFileSync(podfile, patched);
      return cfg;
    },
  ]);

  return config;
};