#!/usr/bin/env bash
set -euo pipefail

mkdir -p .claude/logs
hook_input="$(cat || true)"

if [[ -n "${hook_input}" ]]; then
  printf '%s\n' "${hook_input}" >> .claude/logs/mcp-tool-failures.jsonl
fi

err_msg=""
if command -v jq >/dev/null 2>&1 && [[ -n "${hook_input}" ]]; then
  err_msg="$(printf '%s' "${hook_input}" | jq -r '.error // .message // ""' 2>/dev/null || true)"
fi

if [[ "${err_msg}" == *"int_parsing"* ]] || [[ "${err_msg}" == *"list_type"* ]] || [[ "${err_msg}" == *"tuple_type"* ]]; then
  cat >&2 <<'EOF'
[bid-review-team hook] MCP 参数类型错误，请仅修正类型后重试同一调用：
- page_idx: 3 或 [0,1,2]，不要写成 "[0,1,2]"
- page_range: [3,8]，不要写成 "[3,8]"
- element_types: ["text","table"]，不要写成字符串
EOF
fi
