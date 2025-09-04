#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-origin}"
MAIN="${MAIN:-main}"
PAGES="${PAGES:-reports-pages}"
REPORTS_DIR="reports"
DATA_DIR="data"

require_clean() {
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Working tree not clean. Commit or stash changes first." >&2
    exit 1
  fi
}

# Ensure we are in a git repo and on main
git rev-parse --is-inside-work-tree >/dev/null
git fetch "$REMOTE" --prune
if git show-ref --verify --quiet "refs/heads/$MAIN"; then
  git switch "$MAIN"
elif git show-ref --verify --quiet "refs/remotes/$REMOTE/$MAIN"; then
  git switch -C "$MAIN" "$REMOTE/$MAIN"
else
  echo "Main branch '$MAIN' not found." >&2
  exit 1
fi

require_clean

# Commit current publishing setup to main (if not committed yet)
if ! git diff --quiet -- .gitignore slacgs/publish .github/workflows scripts/publish_output_to_pages.sh 2>/dev/null; then
  git add -A
  git commit -m "chore(publish): add gh-pages index (scenarios+data), CI updater, publish helper; ignore output/ on main"
fi

git push "$REMOTE" "$MAIN"

# Initialize reports-pages if missing
if ! git ls-remote --exit-code --heads "$REMOTE" "$PAGES" >/dev/null 2>&1; then
  echo "Initializing $PAGES branch..."
  git switch --orphan "$PAGES"
  git rm -rf . >/dev/null 2>&1 || true
  rm -rf ./* || true
  mkdir -p "$REPORTS_DIR" "$DATA_DIR"
  touch .nojekyll
  echo "<!doctype html><title>slacgs</title><h1>Publishing…</h1>" > index.html
  git add -A
  git commit -m "chore(publish): initialize reports-pages branch"
  git push -u "$REMOTE" "$PAGES"
  git switch "$MAIN"
fi

# Seed reports-pages from local output
git worktree add reports-pages "$PAGES"
mkdir -p reports-pages/"$REPORTS_DIR" reports-pages/"$DATA_DIR"
if [ -d output/reports ]; then
  cp -R output/reports/. reports-pages/"$REPORTS_DIR"/
else
  echo "Note: output/reports not found; skipping HTML sync."
fi
shopt -s nullglob
jsons=( output/*.json )
if [ ${#jsons[@]} -gt 0 ]; then
  cp output/*.json reports-pages/"$DATA_DIR"/
else
  echo "Note: no JSON files under output/; skipping JSON sync."
fi

touch reports-pages/.nojekyll

pushd reports-pages >/dev/null
if [ -n "$(git status --porcelain)" ]; then
  git add -A
  git commit -m "chore(publish): seed reports and data"
  git push "$REMOTE" "$PAGES"
fi
popd >/dev/null

git worktree remove reports-pages --force || true

echo "Setup complete. In GitHub Settings → Pages, set Source to 'Deploy from a branch', Branch: reports-pages, Folder: root."
