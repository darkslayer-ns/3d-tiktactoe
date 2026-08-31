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
#   make test         backend pytest + mobile jest + C++ parity
#   make run-backend  dev server on :8100
#   make run-frontend dev server on :5173
#   make clean        remove build artifacts

SHELL := /bin/bash
ROOT  := $(CURDIR)
NCPU  := $(shell nproc 2>/dev/null || echo 4)

MODEL_PT  := $(ROOT)/dev/model_universal.pt
MODEL_BIN := $(ROOT)/cpp/model.bin
CPP_BUILD := $(ROOT)/cpp/build
CPP_SRCS  := $(wildcard cpp/src/*.cpp)
CPP_HDRS  := $(wildcard cpp/include/tfm/*.hpp)

.PHONY: all help cpp parity model embed fixtures \
        frontend mobile mobile-typecheck mobile-test mobile-prebuild \
        apk android-debug \
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