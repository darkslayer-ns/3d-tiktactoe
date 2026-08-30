# Neon Cube — React Native (self-contained, no backend)

The 3D tic-tac-toe game rebuilt for iOS/Android. **Everything runs on-device** —
there is no API server. Game rules and the AI search are TypeScript; the neural
network forward pass runs in the same hand-written **C++ engine** the Python
backend uses, compiled into the app and called through JSI.

```
TS (rules + lookahead search + predictor)
  └── JSI ──> libtfm (cpp/ engine, embedded weights) ──> move
```

- **Expo SDK 57** / React Native 0.86 / React 19
- 3D board via `@react-three/fiber/native` + `expo-gl` (three.js on GL)
- Tap to select a cell, **Place** to confirm; drag rotates, pinch zooms
- PvE only, board sizes 3×3×3 / 4×4×4 / 5×5×5, difficulties Easy/Medium/Hard

## Layout

```
mobile-rn/
  src/
    game/board.ts        rules + winning lines (port of backend/game/board.py)
    ai/math.ts           sigmoid / softmax / argmax / sample (pure)
    ai/predictor.ts      opponent predictor (port of backend/ml/predictor.py)
    ai/mover.ts          LookaheadMover + difficulty knobs (port)
    ai/engine.ts         EvalEngine seam (native or mock)
    ai/types.ts          shared contracts
    native/TfmEngine.ts  JS side of the JSI module
    ui/                  Board3D, GameScreen, MenuSheet, StatusBar, theme
    __tests__/           board/mover/predictor/parity tests + fixtures
  native/                C++ JSI module (compiles cpp/ engine + embedded weights)
  plugins/withTfmEngine.js  Expo config plugin (Android CMake + iOS pod)
  scripts/
    embed_weights.py     cpp/model.bin -> native/include/tfm_model_data.h
    gen_parity_fixture.py  generate AI parity fixtures from the Python backend
```

## Build & run (real machine)

```bash
cd mobile-rn
npm install
npm run embed                 # regenerate weights header after re-exporting model.bin
npx expo prebuild             # runs withTfmEngine plugin (writes android/ ios/)
npx expo run:android          # requires Android SDK + NDK
npx expo run:ios              # requires macOS + Xcode + CocoaPods
```

**iOS manual step (once):** the TurboModule must be registered from native
code — move `ReactNativeDelegate` into a `.mm` file overriding
`getTurboModule:jsInvoker:` to return `tfmengine::TfmEngineTurboModule`. Exact
snippet: `native/README.md`.

## Building an APK

Requires a machine with the Android toolchain (this repo's dev box does **not**
have it):

1. **JDK 17+** (RN 0.86 / AGP needs 17; not Java 11)
2. **Android SDK + NDK 27.x**: install `platform-tools`, a platform matching the
   RN 0.86 defaults, `build-tools`, and `ndk;27.1.12297006` (e.g. via Android
   Studio's SDK Manager or `sdkmanager`), then accept licenses.

```bash
export ANDROID_HOME=$HOME/Android/Sdk
export JAVA_HOME=/path/to/jdk17
cd mobile-rn
npm install
npx expo prebuild          # writes android/ + ios/ (runs the withTfmEngine plugin)
cd android && ./gradlew assembleRelease
```

or from the repo root:

```bash
make apk                   # embeds weights + prebuilds + assembleRelease
```

Output:
`mobile-rn/android/app/build/outputs/apk/release/app-release.apk`

Debug (install on a connected device/emulator):

```bash
make android-debug         # == npx expo run:android
```

If you only have `local.properties`, it must point at the SDK:
`sdk.dir=/home/<you>/Android/Sdk`.

## Re-exporting weights

After retraining `cpp/model.bin` (via `cpp/tools/export_weights.py`):

```bash
cd mobile-rn && npm run embed
```

The weights ship inside the binary as a generated C array — no file I/O, no
bundler asset path.

## Tests

```bash
cd mobile-rn && npx jest          # 33 tests: board, mover, predictor, parity
```

The **parity gate** (`src/__tests__/parity.test.ts`) replays fixtures captured
from the real Python backend (same C++ engine, seeded RNG) and asserts the TS
AI chooses the identical moves — proving the on-device game plays exactly like
the server.

## Sharing the engine

The C++ engine (`../cpp/`) is frozen and shared: the Python backend loads it
via `ctypes`, the app compiles the same `cpp/src/*.cpp`. Cross-platform parity
is enforced by `cpp/tools/parity.sh`.