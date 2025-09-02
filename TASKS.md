# Project Tasks

Authoritative checklist for planned work. Use IDs in commits and code TODOs.

## Publishing
- [ ] TASK-001: Set up `gh-pages` branch and GitHub Pages settings
- [ ] TASK-002: Implement index generator (`slacgs/publish/publisher.py`) linking scenario reports
- [ ] TASK-003: Add GitHub Action to publish reports from `main` to `gh-pages`

## Repo Structure
- [ ] TASK-010: Adopt `src/` layout and update `setup.py` (`package_dir={"": "src"}`)
- [ ] TASK-011: Organize modules: `core/`, `reporting/`, `publish/`
- [ ] TASK-012: Switch to absolute imports post-move; update demos/examples
- [ ] TASK-022: Fix `SCENARIOS` handling to avoid overriding full list in `demo.py`

## Public API
- [ ] TASK-020: Make explicit exports in `slacgs/__init__.py`; remove import-time side effects
- [ ] TASK-021: Remove wildcard imports in repo and demos

## Simulator–Report Decoupling
- [ ] TASK-030: Introduce `ReportData` dataclass capturing report inputs
- [ ] TASK-031: Change `Report` to accept `ReportData`; eliminate runtime dependency on `Simulator`
- [ ] TASK-032: Update `Simulator` to build `ReportData` and stop passing `self` into `Report`

## CLI
- [ ] TASK-040: Create `slacgs/cli.py` and `slacgs/__main__.py`; add console entry point `slacgs`
- [ ] TASK-041: Implement subcommands: `run-experiment`, `run-custom-scenario`, `run-custom-simulation`, `add-simulation`, `publish-reports`
- [ ] TASK-042: Support inputs via `--file` JSON, `--params`, interactive prompts; add global flags (`--output-dir`, `--log-level`, `--log-file`, `--yes`)

## Config & Logging
- [ ] TASK-050: Add `slacgs/config.py` with defaults + env + `~/.config/slacgs/config.json`
- [ ] TASK-051: Add `slacgs/logging_utils.py` with rotating file + console handlers

## Automation
- [ ] TASK-060: Regenerate index and publish on updates (script + workflow)

## HTML Polish
- [ ] TASK-070: Add CSS/JS assets; improve layout, tiles, timestamps, responsiveness

## Documentation
- [ ] TASK-080: Update README with CLI usage, Pages URL, and deprecate Google Drive integration

## Deprecations & Cleanup
- [ ] TASK-090: Deprecate Google Drive/Sheets modules and paths; keep legacy but remove from CLI

## Testing & CI
- [ ] TASK-100: Add smoke tests for CLI and publisher; basic CI workflow for tests

## Acceptance Criteria
- [ ] TASK-110: Validate install and CLI (`slacgs -h`), local report generation, index publication to GitHub Pages, and removal of wildcard imports

