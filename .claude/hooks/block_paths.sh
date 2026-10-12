#!/usr/bin/env bash
# PreToolUse hook on Edit|Write (docs/PLAN.md §9.4): the dataset is read-only.
# Add frozen files here once reviewed (configs/paper/claims.yaml, docs/HYPOTHESIS.md).
root="$(git rev-parse --show-toplevel)"
path=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))')
case "$path" in
  "$root"/data/README.md|data/README.md) ;;  # committed documentation, editable
  "$root"/data/*|data/*)
    echo "Blocked: $path is under data/, which is never edited (CLAUDE.md). Regenerate it with scripts/download_data.py." >&2
    exit 2;;
esac
exit 0
