#!/usr/bin/env bash
set -euo pipefail

LOG_DIR=".claude/logs"
LOG_FILE="$LOG_DIR/mcp-tool-failures.jsonl"
mkdir -p "$LOG_DIR"

HOOK_INPUT="$(cat || true)"
if [[ -n "${HOOK_INPUT}" ]]; then
  printf "%s\n" "$HOOK_INPUT" >> "$LOG_FILE"
fi

ERR_MSG=""
if command -v jq >/dev/null 2>&1 && [[ -n "${HOOK_INPUT}" ]]; then
  ERR_MSG="$(printf "%s" "$HOOK_INPUT" | jq -r '.error // .message // ""' 2>/dev/null || true)"
fi

if [[ "${ERR_MSG}" == *"int_parsing"* ]] || [[ "${ERR_MSG}" == *"list_type"* ]] || [[ "${ERR_MSG}" == *"tuple_type"* ]]; then
  cat >&2 <<'EOF'
[bid-review-team hook] MCP 参数类型错误，已记录到 .claude/logs/mcp-tool-failures.jsonl
请重试同一工具调用并修正参数类型：
- page_idx: 3 或 [0,1,2]，不要写成 "[0,1,2]"
- page_range: [3,8]，不要写成 "[3,8]"
- element_types: ["text","table"]，不要写成字符串
EOF
fi

exit 0
