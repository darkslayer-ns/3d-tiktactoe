# Native ExecuTorch engine

This is the production inference path for iOS/Android.

It loads `assets/model.pte` (an ExecuTorch program exported from the trained
PyTorch model) and runs it through the ExecuTorch C++ runtime, exposing a tiny
C API that Flutter calls via `dart:ffi`.

```
Dart (search/game logic)
  └── FFI ──> libttt_engine.so / .dylib / .framework
                  └── ExecuTorch C++ runtime ──> model.pte
```

Why ExecuTorch instead of a hand-written C/Dart transformer?
- The model is exported once from the same PyTorch checkpoint used by the
  backend — no reimplementation, no risk of divergence.
- ExecuTorch is the official PyTorch-on-device runtime: small, fast, and it
  targets the iPhone Neural Engine / Android NNAPI.

## Building

Android:
```
flutter build apk   # Gradle builds the CMake target in android/app/src/main/cpp
```

iOS (requires a Mac + Xcode):
```
flutter build ios
```

The Dart side (`lib/engine/ttt_engine.dart`) exposes `ExecuTorchEngine`,
which loads the FFI symbol `ttt_forward(int* board, int n, double* out)`.

## Re-exporting weights

After retraining, run:
```
python3 scripts/export_pte.py   # writes mobile/assets/model.pte
```
