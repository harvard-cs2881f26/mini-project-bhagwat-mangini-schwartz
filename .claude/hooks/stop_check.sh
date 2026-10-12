#!/usr/bin/env bash
# Stop hook (docs/PLAN.md §9.4): a session can't end with tests, lint or types failing.
# Runs pytest on everything; ruff and pyright only on .py files this branch adds or changes
# (vs origin/main, plus untracked), and `ruff format --check` only on files it adds, so
# existing modules aren't reformatted as a side effect.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 0
PY=.venv/bin/python
base=$(git merge-base HEAD origin/main 2>/dev/null || echo HEAD)
changed=$( { git diff --name-only --diff-filter=AMR "$base" -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | sort -u)
added=$( { git diff --name-only --diff-filter=A "$base" -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | sort -u)

fail=0
out=$("$PY" -m pytest -q 2>&1) || { fail=1; echo "pytest failed:" >&2; echo "$out" | tail -25 >&2; }
if [ -n "$changed" ]; then
  out=$(.venv/bin/ruff check $changed 2>&1) || { fail=1; echo "ruff check failed:" >&2; echo "$out" | tail -25 >&2; }
  out=$(npx --yes pyright --level error $changed 2>&1) || { fail=1; echo "pyright failed:" >&2; echo "$out" | tail -25 >&2; }
fi
if [ -n "$added" ]; then
  out=$(.venv/bin/ruff format --check $added 2>&1) || { fail=1; echo "ruff format --check failed (run: ruff format <file>):" >&2; echo "$out" | tail -10 >&2; }
fi
[ $fail -eq 0 ] && exit 0 || exit 2
