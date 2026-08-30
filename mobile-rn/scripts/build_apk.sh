#!/usr/bin/env bash
# Build, sign, and copy a Neon Cube APK. Encodes the full build flow so you
# never hit the "Metro: unable to load script" / wrong-ABI pitfalls again.
#
# Usage (from anywhere in the repo):
#   bash mobile-rn/scripts/build_apk.sh               # release arm64  -> phones
#   bash mobile-rn/scripts/build_apk.sh --emulator    # release x86_64 -> Android Studio emulator (bundle baked in, no Metro)
#   bash mobile-rn/scripts/build_apk.sh --debug       # debug all-ABI  -> needs `npx expo start` + `adb reverse tcp:8081 tcp:8081`
#   OUT_DIR=/path/to/output bash mobile-rn/scripts/build_apk.sh --emulator
#
# Release APKs embed the JS bundle (run standalone, no Metro).
# Debug APKs do NOT embed it — they load from Metro, hence the live-reload flow.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ANDROID_DIR="$ROOT/mobile-rn/android"
OUT_DIR="${OUT_DIR:-$ROOT/mobile-rn/dist-apk}"
MODE="${1:---release}"

case "$MODE" in
  --emulator) VARIANT="release"; ARCH="x86_64"; LABEL="emulator" ;;
  --debug)    VARIANT="debug";   ARCH="";       LABEL="debug" ;;
  --release)  VARIANT="release"; ARCH="arm64-v8a"; LABEL="phone" ;;
  *) echo "unknown mode: $MODE (--release | --emulator | --debug)"; exit 2 ;;
esac

# Android build env (JAVA_HOME, ANDROID_HOME, build-tools on PATH)
if [ -z "${ANDROID_HOME:-}" ]; then
  if [ -f "$HOME/.android-env.sh" ]; then source "$HOME/.android-env.sh"; else
    echo "ANDROID_HOME not set and no ~/.android-env.sh — configure the SDK first."; exit 1
  fi
fi
export PATH="$ANDROID_HOME/build-tools/36.0.0:$PATH"

cd "$ANDROID_DIR"

echo "== gradlew: assemble$VARIANT (arch=${ARCH:-all}) =="
ARCH_ARGS=()
[ -n "$ARCH" ] && ARCH_ARGS=(-PreactNativeArchitectures="$ARCH")
./gradlew "assemble$VARIANT" "${ARCH_ARGS[@]}"

APK="app/build/outputs/apk/$VARIANT/app-$VARIANT.apk"
mkdir -p "$OUT_DIR"

if [ "$VARIANT" = "release" ]; then
  if [ -f release.keystore ]; then
    echo "== zipalign + sign =="
    zipalign -f 4 "$APK" "$OUT_DIR/.aligned.apk"
    apksigner sign --ks release.keystore --ks-pass pass:neoncube --key-pass pass:neoncube \
      --out "$OUT_DIR/neoncube-${LABEL}-release.apk" "$OUT_DIR/.aligned.apk"
    apksigner verify "$OUT_DIR/neoncube-${LABEL}-release.apk"
    rm -f "$OUT_DIR/.aligned.apk"
  else
    cp "$APK" "$OUT_DIR/neoncube-${LABEL}-release-unsigned.apk"
    echo "NOTE: no release.keystore — copied UNSIGNED (not installable on devices)."
  fi
else
  cp "$APK" "$OUT_DIR/neoncube-debug-all-abi.apk"
fi

echo "== output: $OUT_DIR =="
ls -la "$OUT_DIR" | grep -E "neoncube|total"