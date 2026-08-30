#!/usr/bin/env bash
# One-shot Android toolchain installer for the Neon Cube APK (RN 0.86 / Expo 57).
# Ubuntu/Debian incl. WSL. Run from repo root:
#     bash mobile-rn/scripts/android-setup.sh
# (will prompt for sudo to install the JDK; SDK downloads ~2-3 GB into $ANDROID_HOME)
set -euo pipefail

SDK_ROOT="${ANDROID_HOME:-$HOME/Android/Sdk}"
CMDLINE_ID="11076708"          # cmdline-tools build id (see dl.google.com/android/repository)

echo "==> [1/4] JDK 17 (AGP + Gradle 9 require 17, not the installed 11)"
if ! /usr/lib/jvm/java-17-openjdk-amd64/bin/java -version >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y openjdk-17-jdk unzip wget
fi
JAVA_HOME_17="/usr/lib/jvm/java-17-openjdk-amd64"

echo "==> [2/4] Android command-line tools"
mkdir -p "$SDK_ROOT/cmdline-tools"
if [ ! -x "$SDK_ROOT/cmdline-tools/latest/bin/sdkmanager" ]; then
  curl -fsSL "https://dl.google.com/android/repository/commandlinetools-linux-${CMDLINE_ID}_latest.zip" \
    -o /tmp/cmdline-tools.zip
  rm -rf "$SDK_ROOT/cmdline-tools/latest" "$SDK_ROOT/cmdline-tools/cmdline-tools"
  unzip -q /tmp/cmdline-tools.zip -d "$SDK_ROOT/cmdline-tools"
  mv "$SDK_ROOT/cmdline-tools/cmdline-tools" "$SDK_ROOT/cmdline-tools/latest"
fi

SDKM="$SDK_ROOT/cmdline-tools/latest/bin/sdkmanager"
echo "==> [3/4] Accepting licenses"
yes | "$SDKM" --licenses >/dev/null 2>&1 || true

echo "==> [4/4] Installing SDK packages (RN 0.86 defaults)"
"$SDKM" --install \
  "platform-tools" \
  "platforms;android-36" \
  "build-tools;36.0.0" \
  "ndk;27.1.12297006"

echo
echo "== Done. Add to ~/.bashrc and re-source: =="
cat <<EOF
export ANDROID_HOME=$SDK_ROOT
export ANDROID_SDK_ROOT=$SDK_ROOT
export JAVA_HOME=$JAVA_HOME_17
export PATH=\$ANDROID_HOME/platform-tools:\$PATH
EOF
echo
echo "Then build the APK from the repo root:"
echo "    make apk"
echo "If gradle asks for a different platform/build-tools/NDK version,"
echo "install exactly what it reports, e.g.:"
echo "    $SDKM --install \"platforms;android-XX\" \"ndk;YY.Y.YYYYYY\""