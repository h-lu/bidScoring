#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=".claude/logs"
mkdir -p "$LOG_DIR"

{
  printf "[%s] session_start plugin=bid-review-team\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$LOG_DIR/plugin-hooks.log"

exit 0
