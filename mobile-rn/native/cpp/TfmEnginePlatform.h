#pragma once

// Platform-specific bits of the native module.
//
// The engine's numerical kernels (BLAS: Apple Accelerate vs Android Eigen +
// armv8-a+dotprod) live in the shared core at cpp/src/ops.cpp and are selected
// by each platform's build config (native/CMakeLists.txt for Android,
// native/TfmEngine.podspec for iOS). What remains platform-specific HERE is
// only the background-search thread scheduling.

#if defined(__APPLE__)
#include <pthread.h>
namespace tfmengine {
inline void setBackgroundThreadPriority() {
  // Run the off-JS-thread search at utility QoS so it never starves the
  // UI/render thread.
  pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
}
}  // namespace tfmengine
#else
#include <sys/resource.h>
namespace tfmengine {
inline void setBackgroundThreadPriority() {
  // Android/Linux: nice the background search so clicks/render stay smooth.
  setpriority(PRIO_PROCESS, 0, 10);
}
}  // namespace tfmengine
#endif
