#!/usr/bin/env bash
# PreToolUse hook on Edit|Write (docs/PLAN.md §9.4): the dataset and frozen files are read-only.
# Frozen so far: configs/paper/claims.yaml. Add docs/HYPOTHESIS.md here once it is frozen.
root="$(git rev-parse --show-toplevel)"
path=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("tool_input",{}).get("file_path",""))')
case "$path" in
  "$root"/data/README.md|data/README.md) ;;  # committed documentation, editable
  "$root"/data/*|data/*)
    echo "Blocked: $path is under data/, which is never edited (CLAUDE.md). Regenerate it with scripts/download_data.py." >&2
    exit 2;;
  "$root"/configs/paper/claims.yaml|configs/paper/claims.yaml)
    echo "Blocked: configs/paper/claims.yaml is frozen (Noa reviewed it; WORKLOG 2026-10-12). Edit it by hand only if the paper changes." >&2
    exit 2;;
esac
exit 0
