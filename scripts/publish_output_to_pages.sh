#!/usr/bin/env bash
set -euo pipefail
REMOTE="${REMOTE:-origin}"
PAGES="${PAGES:-gh-pages}"
REPORTS_DIR="reports"
DATA_DIR="data"

git fetch "$REMOTE" --prune
if ! git ls-remote --exit-code --heads "$REMOTE" "$PAGES" >/dev/null 2>&1; then
  echo "Error: '$PAGES' branch not found on remote. Initialize it first." >&2
  exit 1
fi

git worktree add gh-pages "$PAGES"

mkdir -p gh-pages/"$REPORTS_DIR" gh-pages/"$DATA_DIR"

if [ -d output/reports ]; then
  cp -R output/reports/. gh-pages/"$REPORTS_DIR"/
else
  echo "Note: output/reports not found; skipping HTML sync."
fi

shopt -s nullglob
jsons=( output/*.json )
if [ ${#jsons[@]} -gt 0 ]; then
  cp output/*.json gh-pages/"$DATA_DIR"/
else
  echo "Note: no JSON files under output/; skipping JSON sync."
fi

touch gh-pages/.nojekyll

pushd gh-pages >/dev/null
if [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -m "chore(publish): sync output to gh-pages"
  git push "$REMOTE" "$PAGES"
else
  echo "No changes to publish."
fi
popd >/dev/null

git worktree remove gh-pages --force || true
echo "Published. Ensure GitHub Pages is set to branch: gh-pages, folder: /"

