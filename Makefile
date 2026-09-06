# Neon Cube — top-level build orchestration.
#
#   make all          build the C++ engine, web frontend, and verify mobile JS
#   make cpp          build the C++ transformer engine (lib + cli + parity_test)
#   make parity       full C++ <-> PyTorch parity gate (cpp/tools/parity.sh)
#   make model        export model_universal.pt -> cpp/model.bin
#   make embed        model.bin -> mobile-rn weights header (after retraining)
#   make frontend     build the web frontend
#   make mobile       mobile-rn: embed + typecheck + jest (game logic + parity)
#   make mobile-prebuild   mobile-rn: generate android/ ios/ via expo prebuild
#   make apk               build a release APK (needs Android SDK/NDK + JDK 17)
#   make android-debug     build+install debug build on a connected device
#   make ios-internal / ios-release          iOS Release build (device); -internal enables the AI debug toggle
#   make ios-internal-ipa / ios-release-ipa  iOS archive + App Store IPA (distribution)
#   make android-internal / android-release  signed Android release APK (-internal enables the AI debug toggle)
#   make internal / release                  build both platforms (internal / public)
#   make test         backend pytest + mobile jest + C++ parity
#   make run-backend  dev server on :8100
#   make run-frontend dev server on :5173
#   make clean        remove build artifacts

SHELL := /bin/bash
ROOT  := $(CURDIR)
NCPU  := $(shell nproc 2>/dev/null || echo 4)

# Local build secrets (gitignored): IOS_TEAM, KEY_STORE_PASS, KEY_PASS. Values
# use `?=` so a shell-env override wins. See mobile-rn/scripts/build-secrets.env.
-include $(ROOT)/mobile-rn/scripts/build-secrets.env

MODEL_PT  := $(ROOT)/dev/model_universal.pt
MODEL_BIN := $(ROOT)/cpp/model.bin
CPP_BUILD := $(ROOT)/cpp/build
CPP_SRCS  := $(wildcard cpp/src/*.cpp)
CPP_HDRS  := $(wildcard cpp/include/tfm/*.hpp)

.PHONY: all help cpp parity model embed fixtures \
        frontend mobile mobile-typecheck mobile-test mobile-prebuild \
        apk android-debug \
        ios-internal ios-release ios-internal-ipa ios-release-ipa \
        android-internal android-release internal release \
        backend-test test run-backend run-frontend clean

all: cpp frontend mobile-typecheck mobile-test backend-test
	@echo
	@echo "== all targets done =="

help:
	@grep -E '^#   make ' $(MAKEFILE_LIST) | sed 's/^#   //'

# ---------------------------------------------------------------------------
# C++ engine
# ---------------------------------------------------------------------------

cpp: $(CPP_BUILD)/parity_test $(CPP_BUILD)/tfm-cli $(CPP_BUILD)/libmodel.so

$(CPP_BUILD)/parity_test: $(CPP_SRCS) $(CPP_HDRS) cpp/CMakeLists.txt
	cmake -S cpp -B $(CPP_BUILD) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(CPP_BUILD) -j$(NCPU)

$(CPP_BUILD)/tfm-cli: $(CPP_BUILD)/parity_test

$(CPP_BUILD)/libmodel.so: $(CPP_BUILD)/parity_test

model: $(MODEL_BIN)

$(MODEL_BIN): $(MODEL_PT)
	python3 cpp/tools/export_weights.py $(MODEL_PT) $(MODEL_BIN)

fixtures: $(MODEL_BIN)
	python3 cpp/tools/make_fixtures.py $(MODEL_PT) cpp/fixtures

# Full C++ <-> PyTorch parity: export weights, generate fixtures, compile,
# run the golden fixture test + the randomized ctypes cross-check.
parity: cpp
	bash cpp/tools/parity.sh $(MODEL_PT)

# ---------------------------------------------------------------------------
# Web frontend
# ---------------------------------------------------------------------------

frontend:
	cd dev/frontend && npm install --no-audit --no-fund && npm run build

# ---------------------------------------------------------------------------
# Mobile (React Native)
# ---------------------------------------------------------------------------

mobile: embed mobile-typecheck mobile-test

# cpp/model.bin -> mobile-rn/native/include/tfm_model_data.h
embed: $(MODEL_BIN)
	cd mobile-rn && python3 scripts/embed_weights.py $(MODEL_BIN)

mobile-typecheck:
	cd mobile-rn && npx tsc --noEmit

mobile-test:
	cd mobile-rn && npx jest

# Generate android/ and ios/ native projects (runs the withTfmEngine plugin).
# Device builds afterwards: `expo run:android` / `expo run:ios`.
mobile-prebuild:
	cd mobile-rn && npx expo prebuild

# ---------------------------------------------------------------------------
# Android APK
# ---------------------------------------------------------------------------
# Prereqs (one-time, on the build machine):
#   JDK 17+            export JAVA_HOME=...
#   Android SDK        export ANDROID_HOME=$HOME/Android/Sdk   (SDK cmdline-tools
#                      installed: platform + build-tools matching RN 0.86 defaults,
#                      NDK 27.x, plus accepted licenses)
#   local.properties   OR create mobile-rn/android/local.properties:
#                          sdk.dir=$HOME/Android/Sdk

# Release APK -> mobile-rn/android/app/build/outputs/apk/release/app-release.apk
apk: mobile-prebuild embed
	@if [ -z "$$ANDROID_HOME" ] && [ -z "$$ANDROID_SDK_ROOT" ]; then \
	  echo "ERROR: ANDROID_HOME not set. Install the Android SDK + NDK first."; \
	  echo "  export ANDROID_HOME=$$HOME/Android/Sdk"; exit 1; fi
	@if ! echo "$$JAVA_HOME" | grep -q .; then \
	  echo "WARNING: JAVA_HOME not set (needs JDK 17+)."; fi
	cd mobile-rn/android && ./gradlew assembleRelease

# Debug build + install on a connected device/emulator (adb)
android-debug: mobile-prebuild embed
	cd mobile-rn && npx expo run:android

# ---------------------------------------------------------------------------
# Release builds (embedded JS bundle — no Metro / dev server)
# ---------------------------------------------------------------------------
#   ios-internal / ios-release        iOS .app (Release) for a connected device
#   ios-internal-ipa / ios-release-ipa  iOS archive + App Store IPA (distribution)
#   android-internal / android-release  signed Android release APK
#   internal / release                build both platforms
#
# "Internal" builds bake EXPO_PUBLIC_INTERNAL_DEBUG=1 so the AI debug toggle /
# model-knowledge panel are included; public release builds compile them out.
# Both are Release configs (embedded bundle, no Metro).
#
# Prereqs (one-time):
#   - run `npx expo prebuild` + `cd ios && pod install` once (never auto-run
#     here: prebuild wipes android/release.keystore)
#   - mobile-rn/android/release.keystore must exist (build_apk.sh signs with it)
#   - iOS: a device must be connected (or set IOS_UDID=<udid>)
#   - Android: ANDROID_HOME / JAVA_HOME set (build_apk.sh sources ~/.android-env.sh)

IOS_UDID ?= $(shell xcrun xctrace list devices 2>/dev/null | grep -Eo '00008140-[0-9A-F]{16}' | head -1)
IOS_DD   ?= /tmp/isocube-build

define build-ios
	@if [ -z "$(IOS_TEAM)" ]; then \
	  echo "IOS_TEAM unset — put it in mobile-rn/scripts/build-secrets.env"; exit 1; fi
	@if [ -z "$(IOS_UDID)" ]; then \
	  echo "No iOS device found. Connect one, or set IOS_UDID=..."; exit 1; fi
	cd mobile-rn && EXPO_PUBLIC_INTERNAL_DEBUG=$(1) xcodebuild \
		-workspace ios/ISOCUBE.xcworkspace -scheme ISOCUBE \
		-configuration Release -destination 'id=$(IOS_UDID)' \
		-derivedDataPath '$(IOS_DD)' \
		-allowProvisioningUpdates -allowProvisioningDeviceRegistration \
		build DEVELOPMENT_TEAM=$(IOS_TEAM) CODE_SIGN_STYLE=Automatic
endef

define archive-ios
	@if [ -z "$(IOS_TEAM)" ]; then \
	  echo "IOS_TEAM unset — put it in mobile-rn/scripts/build-secrets.env"; exit 1; fi
	@bash mobile-rn/scripts/make_export_plist.sh /tmp/ISOCUBE-$(2)-export.plist
	cd mobile-rn && EXPO_PUBLIC_INTERNAL_DEBUG=$(1) xcodebuild \
		-workspace ios/ISOCUBE.xcworkspace -scheme ISOCUBE \
		-configuration Release -destination 'generic/platform=iOS' \
		-archivePath '/tmp/ISOCUBE-$(2).xcarchive' \
		-allowProvisioningUpdates \
		archive DEVELOPMENT_TEAM=$(IOS_TEAM) CODE_SIGN_STYLE=Automatic
	xcodebuild -exportArchive \
		-archivePath '/tmp/ISOCUBE-$(2).xcarchive' \
		-exportPath '/tmp/ISOCUBE-$(2)-export' \
		-exportOptionsPlist '/tmp/ISOCUBE-$(2)-export.plist' \
		-allowProvisioningUpdates
	@echo "IPA: /tmp/ISOCUBE-$(2)-export/ISOCUBE.ipa"
endef

ios-internal:
	$(call build-ios,1)

ios-release:
	$(call build-ios,)

ios-internal-ipa:
	$(call archive-ios,1,internal)

ios-release-ipa:
	$(call archive-ios,,release)

android-internal:
	cd mobile-rn && EXPO_PUBLIC_INTERNAL_DEBUG=1 bash scripts/build_apk.sh --release

android-release:
	cd mobile-rn && bash scripts/build_apk.sh --release

internal: ios-internal android-internal
	@echo
	@echo "== internal builds done =="

release: ios-release android-release
	@echo
	@echo "== release builds done =="

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

backend-test:
	cd dev && python3 -m pytest backend/tests -q

test: parity backend-test mobile-test
	@echo
	@echo "== all tests passed =="

# ---------------------------------------------------------------------------
# Dev servers
# ---------------------------------------------------------------------------

run-backend:
	cd dev && bash scripts/dev.sh backend

run-frontend:
	cd dev && bash scripts/dev.sh frontend

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

clean:
	rm -rf $(CPP_BUILD) cpp/fixtures
	cd dev/frontend && rm -rf dist
	cd mobile-rn && rm -rf dist
	@echo "cleaned build artifacts (kept cpp/model.bin and generated headers)"