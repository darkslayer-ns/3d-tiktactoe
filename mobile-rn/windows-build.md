# Building the APK from Windows (Android Studio + WSL)

The repo lives in WSL but builds with **Android Studio on Windows** — no WSL
Android toolchain needed. Windows reaches the WSL filesystem at
`\\wsl.localhost\Ubuntu\...` (distro name is `Ubuntu`).

## One-time

1. Ensure `mobile-rn/` is prebuilt and weights are embedded (already done; re-run
   after retraining):
   ```
   cd /home/darkslayer/3d_tiktactoe
   npx expo prebuild   # under mobile-rn/
   npm run embed       # under mobile-rn/  (regenerates native/include/tfm_model_data.h)
   ```
2. Android Studio → **File → Open**:
   ```
   \\wsl.localhost\Ubuntu\home\darkslayer\3d_tiktactoe\mobile-rn\android
   ```
3. Let Gradle sync. Use the SDK Manager to install whatever AGP asks for:
   - `platforms;android-36`, `build-tools;36.0.0`
   - `ndk;27.1.12297006`  ← required (C++ engine + JSI compile via CMake)
   - JDK 17+ is bundled with Android Studio.

## Build

- **Debug APK** (no signing, installable):
  **Build → Build App Bundle(s)/APK(s) → Build APK(s)**
  → `mobile-rn/android/app/build/outputs/apk/debug/app-debug.apk`
- **Release APK**:
  **Build → Generate Signed Bundle/APK → APK** (create a keystore first).
- **Run on device**: enable USB debugging, plug in, select device, **Run ▶**.

## Notes

- All native paths are relative, so the build resolves `cpp/` (the shared engine)
  and the embedded weights automatically through the WSL mount.
- Don't copy just `mobile-rn/` to Windows — the native CMake needs `cpp/`.
  Either build in place (above) or copy the **whole repo** to `C:\`.
- First sync downloads Gradle 9.3.1 + deps + NDK (~2–3 GB) and is slower over
  the `\\wsl.localhost` mount (9P). Copying the repo to Windows is the
  fallback if it's too slow.