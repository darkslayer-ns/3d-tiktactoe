#!/usr/bin/env bash
# Writes an iOS export options plist (App Store Connect) to "$1", injecting the
# signing team from the gitignored build-secrets.env — the committed repo never
# contains the team ID.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/mobile-rn/scripts/build-secrets.env" 2>/dev/null || true
: "${IOS_TEAM:?IOS_TEAM unset — add it to mobile-rn/scripts/build-secrets.env}"

cat > "$1" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>method</key>
	<string>app-store-connect</string>
	<key>teamID</key>
	<string>${IOS_TEAM}</string>
	<key>uploadSymbols</key>
	<true/>
	<key>compileBitcode</key>
	<false/>
	<key>signingStyle</key>
	<string>automatic</string>
</dict>
</plist>
EOF