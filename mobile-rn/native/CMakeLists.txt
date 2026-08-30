# TfmEngine — JSI host module (Android).
#
# Compiles the shared transformer engine (cpp/src/{ops,layers,model,weights}.cpp)
# plus the JSI glue (native/cpp/TfmEngine.cpp) into a static library that the
# app links into `libappmodules.so` (New Architecture).
#
# Two ways to consume this file:
#
# 1. (Preferred) From the app's own CMake, via the config plugin
#    (plugins/withTfmEngine.js). The plugin writes
#    `android/app/src/main/jni/CMakeLists.txt` which does:
#
#        cmake_minimum_required(VERSION 3.13)
#        project(appmodules)
#        include(${REACT_ANDROID_DIR}/cmake-utils/ReactNative-application.cmake)
#        add_subdirectory(${PROJECT_ROOT_DIR}/../native native_tfm_engine)
#        target_link_libraries(appmodules tfm_engine)
#
#    The RN gradle plugin passes -DREACT_ANDROID_DIR and -DPROJECT_ROOT_DIR to
#    CMake, and ReactNative-application.cmake sets REACT_COMMON_DIR, so no
#    extra configuration is needed here.
#
# 2. Standalone (sanity check only, not needed for the app build):
#        cmake -S native -B native/build \
#              -DREACT_NATIVE_DIR=/path/to/node_modules/react-native
#        cmake --build native/build

cmake_minimum_required(VERSION 3.13)
project(tfm_engine CXX)

# Engine sources are plain C++17. The JSI glue (TfmEngine.cpp) includes RN's
# <react/bridging/*> headers which require C++20, so that single TU is bumped
# below. Never enable -ffast-math/-Ofast: engine parity with PyTorch relies on
# IEEE-754 semantics.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Repository layout -------------------------------------------------------
#   <repo>/cpp          the shared transformer engine (frozen, read-only)
#   <repo>/mobile-rn/native  this module
# CMAKE_CURRENT_SOURCE_DIR is <repo>/mobile-rn/native.
get_filename_component(TFM_REPO_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)

# --- React Native headers (JSI) ---------------------------------------------
# When driven by the app's libappmodules build, REACT_COMMON_DIR is set by
# ReactNative-application.cmake and REACT_ANDROID_DIR by the RN gradle plugin.
# Fall back to a literal REACT_NATIVE_DIR for standalone builds.
if(DEFINED REACT_COMMON_DIR AND NOT REACT_COMMON_DIR STREQUAL "")
  set(TFM_RN_COMMON_DIR "${REACT_COMMON_DIR}")
elseif(DEFINED REACT_ANDROID_DIR AND NOT REACT_ANDROID_DIR STREQUAL "")
  get_filename_component(TFM_RN_COMMON_DIR "${REACT_ANDROID_DIR}/../ReactCommon" ABSOLUTE)
else()
  if(NOT DEFINED REACT_NATIVE_DIR OR REACT_NATIVE_DIR STREQUAL "")
    message(FATAL_ERROR
      "REACT_NATIVE_DIR not set. Either add this project with the app's CMake "
      "(recommended) or pass -DREACT_NATIVE_DIR=<react-native dir>.")
  endif()
  set(TFM_RN_COMMON_DIR "${REACT_NATIVE_DIR}/ReactCommon")
endif()

# --- Target ------------------------------------------------------------------
add_library(tfm_engine STATIC
  "${TFM_REPO_ROOT}/cpp/src/ops.cpp"
  "${TFM_REPO_ROOT}/cpp/src/layers.cpp"
  "${TFM_REPO_ROOT}/cpp/src/model.cpp"
  "${TFM_REPO_ROOT}/cpp/src/weights.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/TfmEngine.cpp"
)

# Public headers: engine headers (cpp/include), embedded weights + in-memory
# loader (native/include), and the JSI glue header (native/cpp). Consumers that
# link tfm_engine (e.g. the app's OnLoad.cpp) get these automatically.
target_include_directories(tfm_engine PUBLIC
  "${TFM_REPO_ROOT}/cpp/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
)

# JSI / TurboModule headers (private to the glue TU).
target_include_directories(tfm_engine PRIVATE
  "${TFM_RN_COMMON_DIR}"
  "${TFM_RN_COMMON_DIR}/callinvoker"
  "${TFM_RN_COMMON_DIR}/jsi"
)

# Engine sources: C++17, -fPIC, no fast-math.
target_compile_options(tfm_engine PRIVATE -fPIC -O2 -fexceptions -frtti)
set_target_properties(tfm_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

# TfmEngine.cpp pulls in RN bridging headers that need C++20.
set_source_files_properties(
  "${CMAKE_CURRENT_SOURCE_DIR}/cpp/TfmEngine.cpp"
  PROPERTIES COMPILE_OPTIONS "-std=c++20"
)

# erf/exp/sqrt live in libm on glibc; harmless on Android/iOS.
if(NOT ANDROID AND NOT APPLE)
  target_link_libraries(tfm_engine PUBLIC m)
endif()