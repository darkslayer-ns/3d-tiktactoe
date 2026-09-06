# ISOCUBE — the React Native app (self-contained, no backend)

Part of the [ISOCUBE project](../README.md): a 3D tic-tac-toe game whose AI is
a small transformer, trained with supervised distillation + self-play RL and
ported to C++. This directory is the **shipping iOS/Android app** — everything
runs on-device, there is no API server. Game rules and the AI search are
TypeScript; the neural-network forward pass runs in the same hand-written
**C++ engine** (`cpp/`) compiled into the app and called through JSI.

```
TS (rules + lookahead search + predictor)
  └── JSI ──> libtfm (cpp/ engine, embedded weights) ──> move
```

- **Expo SDK 57** / React Native 0.86 / React 19
- 3D board via `@react-three/fiber/native` + `expo-gl` (three.js on GL), with
  low-poly assets authored in Blender (`src/three/`)
- Tap to select a cell, **Place** to confirm; drag rotates, pinch zooms
- PvE only, board sizes 3×3×3 / 4×4×4 / 5×5×5 / 6×6×6, Easy/Medium/Hard

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
    three/               Blender-baked geometry (models.ts, geometry.ts)
    __tests__/           board/mover/predictor/parity tests + fixtures
  assets/
    models/              Blender .glb sources + the rendered trophy.png
    sounds/              SFX (.m4a)
  native/                C++ JSI module (compiles cpp/ engine + embedded weights)
  plugins/withTfmEngine.js  Expo config plugin (Android CMake + iOS pod)
  scripts/
    embed_weights.py     cpp/model.bin -> native/include/tfm_model_data.h
    gen_parity_fixture.py  generate AI parity fixtures from the Python backend
    gen_models.mjs       assets/models/*.glb -> src/three/models.ts
```

## Build & run (real machine)

Requires: Node, and for native builds **macOS + Xcode + CocoaPods** (iOS) or
**Android SDK + NDK 27.1.12297006 + JDK 17** (Android).

```bash
cd mobile-rn
npm install
npm run embed                 # regenerate the weights header after re-exporting model.bin
npx expo prebuild             # runs withTfmEngine plugin (writes android/ ios/)
npx expo run:android          # requires Android SDK + NDK
npx expo run:ios              # requires macOS + Xcode + CocoaPods
```

**iOS registration (one manual step after `expo prebuild`):** add
`native/TfmEngineRegistration.mm` to the `ISOCUBE` target in Xcode. It registers
the `TfmEngine` C++ TurboModule via the global module map — no AppDelegate
changes needed. (A prebuilt `ios/` from this repo already includes it at
`ios/ISOCUBE/TfmEngineRegistration.mm`.) Exact details: `native/README.md`.

## Building an APK

Requires a machine with the Android toolchain (JDK 17+, Android SDK, and
**NDK 27.1.12297006**):

```bash
export ANDROID_HOME=$HOME/Android/Sdk
export JAVA_HOME=/path/to/jdk17
cd mobile-rn
npm install
npx expo prebuild          # writes android/ + ios/ (runs the withTfmEngine plugin)
cd android && ./gradlew assembleRelease
```

or, from the repo root:

```bash
make apk                   # embeds weights + prebuilds + assembleRelease
bash mobile-rn/scripts/build_apk.sh --release   # signed arm64 APK
```

Outputs:
`mobile-rn/android/app/build/outputs/apk/release/app-release.apk` (unsigned) and
`mobile-rn/dist-apk/neoncube-phone-release.apk` (signed).

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

## Re-generating the 3D geometry

The low-poly assets are authored in Blender and baked to TypeScript (no
runtime GLB loading):

```bash
node scripts/gen_models.mjs     # assets/models/*.glb -> src/three/models.ts
```

## Tests

```bash
cd mobile-rn && npx jest          # board, mover, predictor, sfx, parity
```

The **parity gate** (`src/__tests__/parity.test.ts`) replays fixtures captured
from the real Python backend (same C++ engine, seeded RNG) and asserts the TS
AI chooses the identical moves — proving the on-device game plays exactly like
the server.

## License

GNU GPL v3 — see [LICENSE](../LICENSE).