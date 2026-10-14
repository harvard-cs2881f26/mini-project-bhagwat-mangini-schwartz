#!/usr/bin/env bash
# PostToolUse hook on Edit|Write: remember which files this session edited, so the Stop hook
# checks only those (other sessions in the same folder aren't blocked by in-progress files).
# Edits made through Bash (sed, heredocs) aren't seen here.
root="$(git rev-parse --show-toplevel)" || exit 0
mkdir -p "$root/.claude/hooks/state"
python3 -c '
import json, sys
d = json.load(sys.stdin)
sid, path = d.get("session_id", ""), d.get("tool_input", {}).get("file_path", "")
if sid and path:
    with open(sys.argv[1] + "/" + sid + ".files", "a") as f:
        f.write(path + "\n")
' "$root/.claude/hooks/state"
exit 0
