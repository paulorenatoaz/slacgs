# Project Tasks

Authoritative checklist for planned work. Use IDs in commits and code TODOs.

- ## Publishing
- [x] TASK-001: Set up GitHub Pages branch and settings
  - [x] Decide branch name: use `reports-pages`
  - [x] Ensure `.nojekyll` at branch root
  - [x] Seed initial `index.html` via publisher (placeholder ok)
  - [x] Set GitHub Pages Source: branch `reports-pages`, folder `/`
  - [x] Verify `scripts/setup_pages.sh` once (one-time init)
- [ ] TASK-002: Implement index generator (`slacgs/publish/publisher.py`) linking scenario reports (in progress)
  - [x] Discovery patterns for scenario reports and JSON
  - [x] Parameterize `--reports-dir`, `--data-dir`, `--site-dir`, `--title`
  - [ ] Improve empty-state UI text when no items found
  - [ ] Add basic tests for discovery functions
  - [x] Support `python -m slacgs.publish.publisher`
  - [ ] Align CSV export paths to `data/` and keep embedding in HTML
  - [ ] Optional: add `--publish` flag to call `scripts/publish_output_to_pages.sh` after index generation
- [ ] TASK-003: Add GitHub Action to publish reports from `main` to `reports-pages`
  - [ ] Build or collect outputs under `output/`
  - [ ] Run publisher to write `index.html` into worktree root
  - [ ] Commit and push to `reports-pages`
  - [ ] Configure branch protection/caching as needed

## Repo Structure
- [ ] TASK-010: Adopt `src/` layout and update `setup.py`
  - [ ] Move `slacgs/` → `src/slacgs/`
  - [ ] Update `setup.py` with `package_dir={"": "src"}` and `packages=find_packages("src")`
  - [ ] Adjust scripts/tests to new layout if needed
- [ ] TASK-011: Organize modules into `core/`, `reporting/`, `publish/`
  - [ ] `core/`: `model.py`, `simulator.py`, `enumtypes.py`, core utils
  - [ ] `reporting/`: `report.py`, `report_utils.py`
  - [ ] Keep `publish/` for publisher and helpers
  - [ ] Move Google Drive/Sheets into `legacy/` (kept for compatibility)
- [ ] TASK-012: Switch to absolute imports post-move; update demos/examples
  - [ ] Replace relative imports with absolute package imports
  - [ ] Update demo scripts and tests to new paths
- [ ] TASK-022: Fix `SCENARIOS` handling in `demo.py`
  - [ ] Remove any override that replaces full `SCENARIOS`
  - [ ] Add opt-in filtering via env (e.g., `SLACGS_SCENARIO_INDEX`) or CLI
  - [ ] Add a test ensuring default keeps full list

## Public API
- [ ] TASK-020: Explicit exports; no import-time side effects
  - [ ] Define `__all__` in `slacgs/__init__.py` for stable surface
  - [ ] Avoid running logic at import-time
  - [ ] Document public API in README/docs
- [ ] TASK-021: Remove wildcard imports repo-wide
  - [ ] Replace `from x import *` with explicit imports
  - [ ] Grep/lint to verify none remain

## Simulator–Report Decoupling
- [ ] TASK-030: Introduce `ReportData` dataclass
  - [ ] Capture inputs, metrics, images, metadata
  - [ ] Optional helpers in `report_utils.py`
- [ ] TASK-031: Refactor `Report` to accept `ReportData`
  - [ ] Drop direct `Simulator` dependency
  - [ ] Move computations into `Simulator`/helpers
- [ ] TASK-032: Update `Simulator` to produce `ReportData`
  - [ ] Stop passing `self` into `Report`
  - [ ] Update demos/tests accordingly

## CLI
- [ ] TASK-040: CLI entry points
  - [ ] Create `slacgs/cli.py` with `main()`
  - [ ] Create `slacgs/__main__.py` to call CLI
  - [ ] Add console entry `slacgs` in `setup.py`
- [ ] TASK-041: Implement subcommands
  - [ ] `run-experiment`, `run-custom-scenario`, `run-custom-simulation`
  - [ ] `add-simulation`, `publish-reports`
- [ ] TASK-042: Inputs and global flags
  - [ ] Support `--file` JSON and `--params` inline
  - [ ] Interactive prompts; respect `--yes`
  - [ ] Global: `--output-dir`, `--log-level`, `--log-file`, `--yes`

## Config & Logging
- [ ] TASK-050: Config module
  - [ ] Defaults + env + `~/.config/slacgs/config.json`
  - [ ] Merge precedence: env > user file > defaults
- [ ] TASK-051: Logging utilities
  - [ ] Rotating file + console handlers
  - [ ] Respect CLI flags; include run metadata

## Automation
- [ ] TASK-060: Auto-regenerate and publish
  - [ ] Script to regenerate index when outputs change
  - [ ] GitHub workflow: on push to `main`, publish to `reports-pages`

## HTML Polish
- [ ] TASK-070: Assets and layout polish
  - [ ] Extract inline CSS to `assets/` and copy in publisher
  - [ ] Improve tiles, responsiveness, timestamps/badges
  - [ ] CSV UX: link downloadable CSVs from report pages and index
  - [ ] Optional: JS CSV viewer to render tables from CSV (defer embedding)
  - [ ] Option to toggle embed vs. load-from-CSV for large tables

## Documentation
- [ ] TASK-080: Update README/docs
  - [ ] Install + CLI usage
  - [ ] Pages URL and publishing instructions
  - [ ] Deprecate Google Drive integration (legacy note)

## Deprecations & Cleanup
- [ ] TASK-090: Deprecate Google Drive/Sheets
  - [ ] Mark modules deprecated; keep legacy path
  - [ ] Remove from CLI; add runtime warnings
  - [ ] Update demos to local-only flows

## Testing & CI
- [ ] TASK-100: Tests and CI
  - [ ] Smoke tests for CLI subcommands and publisher
  - [ ] CI: run tests on 3.9–3.11

## Acceptance Criteria
- [ ] TASK-110: Validate end-to-end
  - [ ] `pip install -e .` and `slacgs -h` works
  - [ ] Local report generation produces HTML/JSON under `output/`
  - [ ] Publisher generates index; Pages serves from `reports-pages`
  - [ ] No wildcard imports; explicit exports only
