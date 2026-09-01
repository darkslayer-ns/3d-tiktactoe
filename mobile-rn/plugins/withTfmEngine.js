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
  withXcodeProject,
  IOSConfig,
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
// iOS: TurboModule registration + engine wiring in the generated Xcode project
// ---------------------------------------------------------------------------
//
// The app runs bridgeless, and Expo SDK 52+'s Swift delegate cannot be
// subclassed from ObjC++, so we register the C++ TurboModule in RN's global
// CXX module map from a static initializer (TfmEngineRegistration.mm). That
// lets `globalThis.__turboModuleProxy` / the bridgeless `nativeModuleProxy`
// resolve `TfmEngine` and install `globalThis.__TfmEngine` (see
// src/native/TfmEngine.ts).
//
// We also compile the shared engine sources directly into the app target
// (CocoaPods sandboxes pods to their own dir, so the pod can't see ../cpp),
// with -O2 (Debug builds default to -O0, making the AI ~10x slower).
//
// The iOS folder is gitignored / regenerated by `expo prebuild`, so all of
// this is recreated here.

// Written to ios/<App>/TfmEngineRegistration.mm.
const IOS_REGISTRATION_SOURCE = `#include <ReactCommon/CallInvoker.h>
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
`;

const IOS_ENGINE_SOURCES = ['model.cpp', 'ops.cpp', 'layers.cpp', 'weights.cpp'];

// Header paths the app target needs to #import <TfmEngine.h> and the
// react/nativemodule headers from the node_modules ReactCommon tree.
// `$(inherited)` MUST stay first so the Pods xcconfig header paths (Expo,
// React-Core, ...) keep flowing in.
const IOS_HEADER_SEARCH_PATHS = [
  '"$(inherited)"',
  '"$(SRCROOT)/../native/cpp"',
  '"$(SRCROOT)/../native/include"',
  '"$(SRCROOT)/../../cpp/include"',
  '"$(SRCROOT)/../node_modules/react-native/ReactCommon"',
];

const COMMENT_KEY = /^_comment$/;

// Writes ios/<App>/TfmEngineRegistration.mm.
function addIosRegistrationFile(config) {
  const iosRoot = config.modRequest.platformProjectRoot; // <mobile-rn>/ios
  const appDir = config.modRequest.projectName; // e.g. "NeonCube"
  const file = path.join(iosRoot, appDir, 'TfmEngineRegistration.mm');
  if (!fs.existsSync(file)) {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(file, IOS_REGISTRATION_SOURCE);
  }
  return config;
}

// Copies the ISOCUBE logo into the generated AppIcon.appiconset (1024px).
function addIosIcon(config) {
  const iosRoot = config.modRequest.platformProjectRoot;
  const appDir = config.modRequest.projectName;
  const iconSet = path.join(
    iosRoot,
    appDir,
    'Images.xcassets',
    'AppIcon.appiconset',
    'App-Icon-1024x1024@1x.png',
  );
  const logo = path.join(config.modRequest.projectRoot, 'assets', 'isocube_icon.png');
  if (fs.existsSync(logo)) {
    fs.mkdirSync(path.dirname(iconSet), { recursive: true });
    fs.copyFileSync(logo, iconSet);
  }
  return config;
}

// Adds the registration source + engine sources to the app target, and sets
// the header search paths.
function patchXcodeProject(xcodeProject, projectName) {
  const appTarget =
    xcodeProject.getTarget('com.apple.product-type.application') ||
    xcodeProject.getFirstTarget();
  if (appTarget) {
    const targetUuid = appTarget.uuid;
    const add = IOSConfig.XcodeUtils.addBuildSourceFileToGroup;

    // Registration file (inside ios/<App>/).
    add({
      filepath: path.join(projectName, 'TfmEngineRegistration.mm'),
      groupName: projectName,
      project: xcodeProject,
      targetUuid,
    });
    setFileRefType(xcodeProject, path.join(projectName, 'TfmEngineRegistration.mm'), 'sourcecode.cpp.objcpp');

    // Engine sources (repo-root cpp/, read-only) with -O2.
    for (const f of IOS_ENGINE_SOURCES) {
      const filepath = path.join('..', '..', 'cpp', 'src', f);
      add({ filepath, groupName: projectName, project: xcodeProject, targetUuid });
      setFileRefType(xcodeProject, filepath, 'sourcecode.cpp.cpp');
      setBuildFileCompilerFlags(xcodeProject, filepath, '-std=c++20 -O2');
    }

    // Header search paths on the app target's build configurations.
    const xcConfigList = xcodeProject.pbxXCConfigurationList();
    const configList = xcConfigList[appTarget.target.buildConfigurationList];
    if (configList && Array.isArray(configList.buildConfigurations)) {
      const sections = xcodeProject.pbxXCBuildConfigurationSection();
      for (const cfg of configList.buildConfigurations) {
        const cfgObj = sections[cfg.value];
        if (!cfgObj || !cfgObj.buildSettings) continue;
        const existing = cfgObj.buildSettings.HEADER_SEARCH_PATHS;
        const list = Array.isArray(existing) ? existing.slice() : existing ? [existing] : [];
        for (const p of IOS_HEADER_SEARCH_PATHS) {
          if (!list.includes(p)) list.push(p);
        }
        cfgObj.buildSettings.HEADER_SEARCH_PATHS = list;
      }
    }

    // Generate dSYMs for the RN prebuilt frameworks (React, ReactNativeDependencies,
    // hermesvm) — they ship without dSYMs, so the archive misses them and App Store
    // Connect symbol upload fails. dsymutil on the built product creates the dSYM
    // next to it; Xcode's archive step then collects it. Idempotent.
    const scriptPhases = (xcodeProject.hash.project.objects.PBXShellScriptBuildPhase) || {};
    let hasDsymPhase = false;
    for (const key of Object.keys(scriptPhases)) {
      if (COMMENT_KEY.test(key)) continue;
      if ((scriptPhases[key].name || '').indexOf('Generate RN prebuilt dSYMs') >= 0) {
        hasDsymPhase = true;
        break;
      }
    }
    if (!hasDsymPhase) {
      const script =
        'set -e; for fw in React ReactNativeDependencies hermesvm; do DEST=$BUILT_PRODUCTS_DIR/$fw.framework.dSYM; ' +
        'if [ ! -d $DEST ]; then BIN=$(find $BUILT_PRODUCTS_DIR -path *$fw.framework/$fw -print -quit 2>/dev/null); ' +
        'if [ -n ${BIN} ]; then dsymutil -o $DEST $BIN >/dev/null 2>&1 || true; fi; fi; done\n';
      xcodeProject.addBuildPhase([], 'PBXShellScriptBuildPhase', 'Generate RN prebuilt dSYMs', targetUuid, {
        shellPath: '/bin/sh',
        shellScript: script,
      });
    }
  }
  return xcodeProject;
}

function unquote(v) {
  return typeof v === 'string' ? v.replace(/^"+|"+$/g, '') : v;
}

// The xcode npm package's pbxFile only knows a few extensions, so .cpp/.mm
// file refs come out as lastKnownFileType = "unknown". Xcode's C++ build rule
// keys off the file type, so set it explicitly.
function setFileRefType(xcodeProject, filepath, lastKnownFileType) {
  const base = path.basename(filepath);
  const refs = xcodeProject.pbxFileReferenceSection();
  for (const key of Object.keys(refs)) {
    if (COMMENT_KEY.test(key)) continue;
    if (unquote(refs[key].path) === filepath || unquote(refs[key].name) === base) {
      refs[key].lastKnownFileType = lastKnownFileType;
      return;
    }
  }
}

// Sets COMPILER_FLAGS on the PBXBuildFile whose fileRef has `path`.
function setBuildFileCompilerFlags(xcodeProject, filepath, flags) {
  const base = path.basename(filepath);
  const refs = xcodeProject.pbxFileReferenceSection();
  let fileRefKey = null;
  for (const key of Object.keys(refs)) {
    if (COMMENT_KEY.test(key)) continue;
    if (unquote(refs[key].path) === filepath || unquote(refs[key].name) === base) {
      fileRefKey = key;
      break;
    }
  }
  if (!fileRefKey) return;
  const buildFiles = xcodeProject.pbxBuildFileSection();
  for (const key of Object.keys(buildFiles)) {
    if (COMMENT_KEY.test(key)) continue;
    if (buildFiles[key].fileRef === fileRefKey) {
      // Pre-quote: the xcode serializer doesn't quote values with spaces, which
      // produces invalid OpenStep plist for `-std=c++20 -O2`.
      buildFiles[key].settings = { COMPILER_FLAGS: JSON.stringify(flags) };
      return;
    }
  }
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

  // iOS: make CocoaPods compile the podspec into the app, write the
  // TurboModule registration source, copy the icon, and wire the engine
  // sources + header paths into the generated Xcode project.
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
  config = withDangerousMod(config, ['ios', addIosRegistrationFile]);
  config = withDangerousMod(config, ['ios', addIosIcon]);
  config = withXcodeProject(config, (cfg) => {
    cfg.modResults = patchXcodeProject(cfg.modResults, cfg.modRequest.projectName);
    return cfg;
  });

  return config;
};