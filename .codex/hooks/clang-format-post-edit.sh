#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
input=$(mktemp)
trap 'rm -f "$input"' EXIT
cat >"$input"

jq -r '
  .tool_input.file_path? // empty,
  (.tool_input.command? // "" |
    split("\n")[] |
    select(test("^\\*\\*\\* (Add|Update) File: |^\\*\\*\\* Move to: ")) |
    sub("^\\*\\*\\* (Add|Update) File: "; "") |
    sub("^\\*\\*\\* Move to: "; "")
  )
' "$input" |
  awk 'NF && /\.(cc|cpp|h|hpp)$/ && !seen[$0]++' |
  while IFS= read -r file; do
    case "$file" in
      /*) path="$file" ;;
      *) path="$repo_root/$file" ;;
    esac

    if [ -f "$path" ]; then
      clang-format -i "$path"
    fi
  done
