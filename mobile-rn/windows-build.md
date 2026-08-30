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

## APK build flow (`scripts/build_apk.sh`)

The whole flow (build → sign → copy) is one command from WSL:

```bash
bash mobile-rn/scripts/build_apk.sh               # release arm64   → real phones
bash mobile-rn/scripts/build_apk.sh --emulator    # release x86_64  → Android Studio emulator
bash mobile-rn/scripts/build_apk.sh --debug       # debug all-ABI   → needs Metro
OUT_DIR=/mnt/c/Users/nikol/Downloads bash mobile-rn/scripts/build_apk.sh --emulator
```

### Key facts to avoid the classic errors

- **"Metro: unable to load script"** = you installed a **debug** APK without a
  running Metro dev server. Debug APKs have no JS bundle. Either run
  `npx expo start` + `adb reverse tcp:8081 tcp:8081`, or use a **release** APK
  (bundle baked in, runs standalone).
- **Architecture matters**: `arm64-v8a` (phone) **will not install** on an
  x86_64 emulator (`INSTALL_FAILED_NO_MATCHING_ABIS`). Use the `--emulator`
  variant for emulators.
- **Signature mismatch**: debug and release APKs are signed with different keys.
  Installing one over the other fails (`INSTALL_FAILED_UPDATE_INCOMPATIBLE`) —
  uninstall the old app first.
- Release signing uses `mobile-rn/android/release.keystore` (password
  `neoncube`). Keep it — it's the app's update identity.