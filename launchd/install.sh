#!/bin/sh
# Install (or reinstall) the launchd agent that keeps `omlxbench run` going.
# Usage: launchd/install.sh            install and start
#        launchd/install.sh uninstall  stop and remove
set -eu

LABEL=com.samalone.omlxbench
PROJECT=$(cd "$(dirname "$0")/.." && pwd)
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
UV=$(command -v uv)
DOMAIN="gui/$(id -u)"

launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
if [ "${1:-}" = "uninstall" ]; then
    rm -f "$PLIST"
    echo "removed $LABEL"
    exit 0
fi

mkdir -p "$PROJECT/data" "$(dirname "$PLIST")"
cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array>
        <string>$UV</string><string>run</string><string>--project</string><string>$PROJECT</string>
        <string>omlxbench</string><string>run</string>
    </array>
    <key>WorkingDirectory</key><string>$PROJECT</string>
    <key>RunAtLoad</key><true/>
    <!-- Restart after a crash, or if the project volume wasn't mounted yet. -->
    <key>KeepAlive</key><true/>
    <key>ThrottleInterval</key><integer>60</integer>
    <!-- Benchmarks are background work. -->
    <key>ProcessType</key><string>Background</string>
    <key>StandardOutPath</key><string>$PROJECT/data/runner.log</string>
    <key>StandardErrorPath</key><string>$PROJECT/data/runner.log</string>
</dict>
</plist>
PLIST

launchctl bootstrap "$DOMAIN" "$PLIST"
echo "installed $LABEL (log: $PROJECT/data/runner.log)"
