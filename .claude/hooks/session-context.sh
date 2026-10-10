#!/usr/bin/env bash
# SessionStart hook: put the competition summary, the latest experiment rows, and the dates of the last
# forum and notebook checks into context.
# Reads only local files; no network.
set -eu
cd "${CLAUDE_PROJECT_DIR:-.}"
[ -f docs/competition.md ] || exit 0
echo "== docs/competition.md (head) =="
head -40 docs/competition.md
echo
echo "== docs/experiments.md (tail) =="
[ -f docs/experiments.md ] && tail -25 docs/experiments.md
echo
echo "== research digests (kaggle-research) =="
for f in docs/discussions.md docs/notebooks.md; do
  [ -f "$f" ] || continue
  d=$(sed -n 's/.*<!-- last-checked: \(.*\) -->.*/\1/p' "$f" | head -1)
  echo "$f last checked: ${d:-no marker; add <!-- last-checked: YYYY-MM-DD -->}"
done
exit 0
