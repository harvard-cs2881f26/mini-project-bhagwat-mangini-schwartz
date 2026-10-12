#!/usr/bin/env bash
# Stop hook (docs/PLAN.md §9.4): a session can't end with tests, lint or types failing on its
# own work. Checks only the .py files this session edited (logged by track_edits.sh); if it
# edited none, nothing runs, so sessions that only read or edit docs aren't blocked.
# ruff check + pyright on those files; `ruff format --check` only on the ones that are new on
# this branch, so existing modules aren't reformatted as a side effect; pytest on everything.
set -uo pipefail
root="$(git rev-parse --show-toplevel)" || exit 0
cd "$root" || exit 0
sid=$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("session_id",""))')
log=".claude/hooks/state/$sid.files"
[ -n "$sid" ] && [ -f "$log" ] || exit 0

files=$(sort -u "$log" | sed "s|^$root/||" | grep -E '\.py$' | while read -r f; do [ -f "$f" ] && echo "$f"; done)
[ -n "$files" ] || exit 0
base=$(git merge-base HEAD origin/main 2>/dev/null || echo HEAD)
new=$(for f in $files; do git cat-file -e "$base:$f" 2>/dev/null || echo "$f"; done)

fail=0
out=$(.venv/bin/python -m pytest -q 2>&1) || { fail=1; echo "pytest failed:" >&2; echo "$out" | tail -25 >&2; }
out=$(.venv/bin/ruff check $files 2>&1) || { fail=1; echo "ruff check failed:" >&2; echo "$out" | tail -25 >&2; }
out=$(npx --yes pyright --level error $files 2>&1) || { fail=1; echo "pyright failed:" >&2; echo "$out" | tail -25 >&2; }
if [ -n "$new" ]; then
  out=$(.venv/bin/ruff format --check $new 2>&1) || { fail=1; echo "ruff format --check failed (run: ruff format <file>):" >&2; echo "$out" | tail -10 >&2; }
fi
[ $fail -eq 0 ] && exit 0 || exit 2
