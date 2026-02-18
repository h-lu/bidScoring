#!/usr/bin/env bash
set -euo pipefail

mkdir -p .claude/logs
printf '[%s] session_start plugin=bid-review-team\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> .claude/logs/plugin-hooks.log
