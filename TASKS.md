# 🚀 SLACGS Pre-Publication Workflow (0.2.0 → 1.0.0)

**Goal:** Polish, test, document, commit, create presentation & video before PyPI release.

## 📊 Execution Roadmap

```
PHASE 1: Simulation Output Polish
    ↓
PHASE 2: Comprehensive Testing
    ↓
PHASE 3: Documentation & Repo Cleanup
    ↓
PHASE 4: GitHub Commit & Tag
    ↓
PHASE 5: Presentation (slides already done ✅)
    ↓
PHASE 6: Video Production & Marketing
    ↓
PHASE 7: PyPI Publication (v1.0.0)
```

## Quick Status
| Phase | Status | Impact |
|-------|--------|--------|
| Console Polish | ⚪ TODO | CLI/UX |
| Testing | ⚪ TODO | Quality |
| Docs | ⚪ TODO | Usability |
| Clean | ⚪ TODO | Repo Health |
| GitHub | ⚪ TODO | Version Control |
| Slides | ✅ DONE | Presentation |
| Video | ⚪ TODO | Marketing |
| PyPI | ⚪ TODO | Release |

---

# 🎨 PHASE 1: Console Output Polish

Implement professional, live-updating console progress display using `rich` library. Replace old-style line-per-iteration logging with a single, in-place updating panel that shows cardinality progress, loss estimates, and convergence status.

## Design Overview

**Key Principle**: Use Rich's live panels that update in-place (max 10 lines on screen) instead of scrolling 100+ iteration lines.

### Architecture
- **Location**: `src/slacgs/progress.py` (standalone module)
- **Core Class**: `ProgressTracker` (already created, needs integration)
- **Integration**: Instantiate in `Simulator.__init__()`, call methods during `Simulator.run()`
- **Dependency**: `rich` (already in setup.py)

### Console Output Stages

#### Stage 1: Simulation Start
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  🚀 SLACGS Simulator Started             ┃
┃  Model params: [1, 4, 0.6]               ┃
┃  Dimensions to analyze: [2, 3]           ┃
┃  Cardinalities: [2, 4, 8, 16, ..., 1024] ┃
┃  Mode: Normal (test_mode=False)          ┃
┃  Output: ~/slacgs/output/                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

#### Stage 2: Cardinality Progress (Live Panel - Updates In-Place)
```
┏ Cardinality Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│ n=16       [████████░░░░░░░░░░░░░░░░░░░░░░] 40% ⏱ 2m 15s ETA  │
│ 🔄 Training SVM (iteration 40/100)...                         │
│                                                               │
│ Loss Estimates @ Checkpoint:                                 │
│   d=2: THEORY=0.4023 | TRAIN=0.3847 | TEST=0.3901 ✓         │
│   d=3: THEORY=0.2156 | TRAIN=0.1945 | TEST=0.2034 ✓         │
│                                                               │
│ Status: d=2 [🟢 improving] d=3 [🟢 improving]                │
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```
- **Updates every checkpoint step** (every 10-20 iterations in test_mode, every 100-200 in full)
- **Panel overwrites itself** (stays at same screen location)
- **Loss colors**:
  - 🟢 Green: Loss improved significantly since last checkpoint
  - 🟡 Yellow: Loss stagnating (convergence threshold reached)
  - 🔵 Blue: Loss value (read-only reference)

#### Stage 3: Completion Summary
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ✅ Simulation Complete                                       ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ 📊 Summary                                                   ┃
┃    Total time:         12m 45s                               ┃
┃    Cardinalities:      10/10 processed                       ┃
┃    Dimensions:         [2, 3] analyzed                       ┃
┃                                                              ┃
┃ 📈 Final Loss Values (Theoretical)                           ┃
┃    d=2: 0.3412 ✓                                             ┃
┃    d=3: 0.1654 ✓                                             ┃
┃                                                              ┃
┃ 📁 Output Locations                                          ┃
┃    Reports:   /home/pr/slacgs/output/reports/               ┃
┃    Data:      /home/pr/slacgs/output/data/                  ┃
┃    Logs:      /home/pr/slacgs/output/logs/                  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

#### Stage 4: Error Handling (On Failure)
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ❌ ERROR: Covariance matrix not positive    ┃
┃          definite                           ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Model params: [1, -2, 0.5]                  ┃
┃                                             ┃
┃ 💡 Hint: Check that all σᵢ > 0 and         ┃
┃          correlations satisfy |ρ| < 1      ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### Verbosity Levels

Support three modes via CLI flags:

```bash
# Default: shows start, checkpoint updates (every 10-20 iters), end summary
slacgs run-simulation --params "[1,1,0.4]" --test-mode

# Debug: adds per-iteration detail logs (old behavior, useful for troubleshooting)
slacgs run-simulation --params "[1,1,0.4]" --test-mode --debug

# Quiet: only start and final summary (CI/batch mode)
slacgs run-simulation --params "[1,1,0.4]" --test-mode --quiet
```

## 1.1 Progress Bars & Spinners
- [x] **POLISH-01** – Add rich progress tracking (COMPLETE 2026-01-14)
  - [x] Install: `pip install rich` (already in setup.py)
  - [x] Create `src/slacgs/progress.py` with ProgressTracker class
  - [ ] Update Simulator.__init__() to instantiate ProgressTracker(verbose, debug)
  - [ ] Update Simulator.run() to call progress methods at key points:
    - `progress.log_simulation_start(model, dims, N, output_path)`
    - `progress.start_cardinality_loop(N_values)`
    - `progress.log_cardinality_start(n, dims, max_iter)`
    - `progress.update_checkpoint(current_iter, loss_sum, iter_N, dims, loss_types)` (every checkpoint)
    - `progress.log_cardinality_complete(n, elapsed_sec)`
    - `progress.log_simulation_complete(total_elapsed, cardinalities_processed, dims)`
    - `progress.finish_cardinality_loop()`
  - [ ] Add --debug flag to CLI (pass to Simulator)
  - [ ] Add --quiet flag to CLI (pass to Simulator)
  - [ ] Test with `slacgs run-experiment --scenarios 1 --test-mode`
  - [ ] Test with `slacgs run-experiment --scenarios 1 --test-mode --debug`

- [ ] **POLISH-02** – Color-coded status messages
  - [ ] Green [✓] for completed steps, improved convergence
  - [ ] Yellow [⚠] for warnings, stagnation
  - [ ] Red [✗] for errors
  - [ ] Blue [ℹ] for informational messages
  - [ ] Use in all progress messages

---

# 🧪 PHASE 2: Comprehensive Testing

Expand test suite to cover core functionality + edge cases.

## 2.1 Unit Tests
- [ ] **TEST-01** – Model class tests (`test/test_model.py`)
  - [ ] Valid 2D/3D/4D parameter vectors
  - [ ] Covariance matrix validation (PD, symmetry)
  - [ ] Invalid parameters (neg sigma, invalid correlations)
  - [ ] Cardinality list generation

- [ ] **TEST-02** – Simulator class tests (`test/test_simulator.py`)
  - [ ] Initialization with various models
  - [ ] Dataset generation reproducibility (with seed)
  - [ ] SVM training per dimensionality
  - [ ] Loss computation (empirical + theoretical)
  - [ ] Stopping criteria behavior
  - [ ] test_mode=True reduces iterations correctly

- [ ] **TEST-03** – Report class tests (`test/test_report.py`)
  - [ ] ReportData creation from Simulator
  - [ ] JSON serialization/deserialization
  - [ ] HTML report generation (no errors, valid HTML)
  - [ ] File I/O (write_to_json, create_html_report)

- [ ] **TEST-04** – CLI tests (`test/test_cli.py`)
  - [ ] run-simulation command (2D, 3D, invalid params)
  - [ ] run-experiment command (predefined, custom, scenarios)
  - [ ] make-report, config, publish commands
  - [ ] --help outputs are consistent
  - [ ] Error handling (missing params, bad JSON)

## 2.2 Integration Tests
- [ ] **TEST-05** – End-to-end workflows (`test/test_integration.py`)
  - [ ] Full simulation → JSON → HTML report
  - [ ] Batch experiment (multiple param sets)
  - [ ] Resume from checkpoint (JSON already exists)
  - [ ] Config loading (env vars, TOML, defaults)
  - [ ] test_mode=True vs False execution time diff

## 2.3 Test Infrastructure
- [ ] **TEST-INFRA-01** – Coverage & CI
  - [ ] Create `pytest.ini` with coverage settings
  - [ ] Target: >80% code coverage
  - [ ] Run: `pytest --cov=slacgs --cov-report=term-missing`
  - [ ] Add .github/workflows/test.yml (Python 3.8–3.12, multi-OS)

---

# 📚 PHASE 3: Documentation & Repo Cleanup

Professional docs and clean repository structure.

## 3.1 Sphinx Documentation
- [ ] **DOC-01** – Regenerate Sphinx docs
  - [ ] `cd docs && make clean && make html`
  - [ ] Verify no warnings/errors
  - [ ] Check docs/_build/html/index.html locally
  - [ ] Decide hosting: GitHub Pages vs Read the Docs vs repo-only

- [ ] **DOC-02** – API Documentation
  - [ ] Ensure all public classes/functions have docstrings (Google style)
  - [ ] Generate docs from docstrings automatically
  - [ ] Add examples to key classes (Model, Simulator, CLI)

## 3.2 README & User Guide
- [ ] **DOC-03** – Update README.md
  - [ ] Add PyPI install: `pip install slacgs`
  - [ ] Add PyPI badge (version, Python support, license)
  - [ ] Add CI/CD badge (from GitHub Actions)
  - [ ] Add quick start: 3-line CLI example
  - [ ] Verify PDF links work (slacgs.pdf, learning_*.pdf)
  - [ ] Add "Citation" section with BibTeX + CFF
  - [ ] Add "Features" and "Architecture" overviews

- [ ] **DOC-04** – Create CONTRIBUTING.md
  - [ ] Development setup (git clone, pip install -e .[dev])
  - [ ] Running tests (pytest commands)
  - [ ] Code style (black, ruff, type hints)
  - [ ] PR process and coding guidelines

## 3.3 Citation & License
- [ ] **DOC-05** – Create CITATION.cff
  - [ ] Include thesis author, year, title
  - [ ] Include institution (UFRJ, IC, CS)
  - [ ] Add preferred citation formats (BibTeX, APA, MLA)
  - [ ] Reference to published paper (if applicable)

## 3.4 Code Quality
- [ ] **CLEAN-01** – Code cleanup
  - [ ] Remove commented-out code blocks
  - [ ] Verify no TODO/FIXME left unaddressed
  - [ ] Run formatters: `black src/ test/`
  - [ ] Run linter: `ruff check src/ test/`
  - [ ] Run type checker: `mypy src/slacgs`

- [ ] **CLEAN-02** – Deprecate legacy code
  - [ ] Add module deprecation warning to demo.py
  - [ ] Update docs: remove demo_scripts references
  - [ ] Verify legacy/ modules have __init__ with warnings
  - [ ] Add note in README: "demo.py is deprecated; use CLI instead"

- [ ] **CLEAN-03** – Final repo hygiene
  - [ ] Verify .gitignore is complete
  - [ ] No unintended files tracked (check git status)
  - [ ] All build artifacts removed (dist/, build/, *.egg-info/)
  - [ ] docs/_build/ not tracked
  - [ ] slacgs.toml in .gitignore (user config, not VCS)

---

# 🔗 PHASE 4: GitHub Commit & Tag

Finalize code and create release tag.

## 4.1 Pre-Commit Checklist
- [ ] **GITHUB-01** – Final review
  - [ ] All tests passing locally: `pytest test/ -v`
  - [ ] All linting & formatting clean: `ruff check && black --check`
  - [ ] No uncommitted changes: `git status` clean
  - [ ] Updated version in:
    - [ ] src/slacgs/__init__.py (__version__ = "0.2.0")
    - [ ] setup.py (version="0.2.0")

## 4.2 Commit & Tag
- [ ] **GITHUB-02** – Commit changes
  - [ ] `git add -A`
  - [ ] `git commit -m "feat: polish output, expand tests, improve docs (v0.2.0)"`
  - [ ] `git push origin main` (or dev branch)

- [ ] **GITHUB-03** – Create release tag
  - [ ] `git tag -a v0.2.0 -m "Release v0.2.0: polished CLI, full test suite, comprehensive docs"`
  - [ ] `git push origin v0.2.0`
  - [ ] Verify tag on GitHub: github.com/youruser/slacgs/releases

---

# 🎤 PHASE 5: Presentation

✅ **ALREADY COMPLETE (2026-01-14)**
- [x] presentation.tex created (18 slides, Beamer)
- [x] presentation_sum.md created (narration + screen recording placeholders)
- [ ] Optional: Polish presentation.tex (theme, fonts, colors)
- [ ] Optional: Add title slide company/affiliation if presenting at event

---

# 🎬 PHASE 6: Video Production

Create a professional 5–10 min video demonstrating SLACGS.

## 6.1 Video Script & Recording
- [ ] **VIDEO-01** – Prepare recording environment
  - [ ] Test screen resolution (1920×1080 recommended)
  - [ ] Test audio quality (microphone, background noise)
  - [ ] Prepare terminal with large font (28–32pt)
  - [ ] Disable notifications

- [ ] **VIDEO-02** – Record screen captures (segments)
  - [ ] Intro (30 sec): title slide, motivation
  - [ ] Theory (1 min): Gaussian classes, error types (minimal)
  - [ ] System overview (1 min): flow diagram
  - [ ] **CLI Demo 1** (2 min): `slacgs run-experiment --scenarios 1 --test-mode`, open HTML report
  - [ ] **CLI Demo 2** (2 min): custom experiment with JSON params file
  - [ ] Library usage (1 min): Python API in Jupyter/script
  - [ ] Roadmap (30 sec): future directions
  - [ ] Outro (30 sec): contact, links

- [ ] **VIDEO-03** – Post-production
  - [ ] Edit segments (DaVinci Resolve, Adobe Premiere, or iMovie)
  - [ ] Add narration (use presentation_sum.md as script)
  - [ ] Add title cards & graphics where needed
  - [ ] Add background music (soft, royalty-free)
  - [ ] Add captions/subtitles (optional but helpful)
  - [ ] Final quality check (audio levels, timing, visual clarity)
  - [ ] Export: MP4, H.264, 1080p, ~10 min duration

## 6.2 Publishing
- [ ] **VIDEO-04** – Upload & distribute
  - [ ] Upload to YouTube (unlisted or public)
  - [ ] Add video link to README.md
  - [ ] Add link to GitHub releases
  - [ ] Tweet/announce on social media (if applicable)
  - [ ] Link in docs / website (slacgs.netlify.app)

---

# 📦 PHASE 7: PyPI Publication (v1.0.0)

Final PyPI release after video is complete.

## 7.1 Pre-Release
- [ ] **PYPI-01** – Build & validate
  - [ ] Install tools: `pip install -U build twine`
  - [ ] Build: `python -m build`
  - [ ] Check: `twine check dist/*`
  - [ ] Test install in clean venv: `pip install dist/slacgs-*.whl`
  - [ ] Verify: `slacgs --help`, `slacgs run-simulation --params "[1,1,0]"`

- [ ] **PYPI-02** – TestPyPI (optional but recommended)
  - [ ] Create account: https://test.pypi.org/account/register/
  - [ ] Configure ~/.pypirc with TestPyPI token
  - [ ] Upload: `twine upload --repository testpypi dist/*`
  - [ ] Test: `pip install --index-url https://test.pypi.org/simple/ slacgs`
  - [ ] Verify CLI works in fresh env

## 7.2 Release
- [ ] **PYPI-03** – Publish to PyPI
  - [ ] Create PyPI account: https://pypi.org/account/register/
  - [ ] Configure ~/.pypirc with PyPI token
  - [ ] Upload: `twine upload dist/*`
  - [ ] Verify on PyPI: https://pypi.org/project/slacgs/
  - [ ] Test: `pip install slacgs` (fresh venv)

- [ ] **PYPI-04** – GitHub Release & Announcement
  - [ ] Tag: `git tag -a v1.0.0 -m "Release v1.0.0: full test suite, polished CLI, professional docs"`
  - [ ] Push: `git push origin v1.0.0`
  - [ ] Create Release on GitHub with release notes + links to:
    - Video
    - PyPI page
    - Documentation
    - Citation info
  - [ ] Announce (Twitter, Reddit, academia channels, etc.)

---

## 📝 Archive: Previous Work (Phases 001–032)

### ✅ Phase 0: Core Architecture (COMPLETE)
Previous 32 tasks (001–032) covering:
- GitHub Pages + publishing workflow
- src/ layout migration
- Config + logging infrastructure
- CLI with typer
- Data architecture (ReportData, circular dep removal)

**Status:** 100% complete as of 2025-12-31

### ✅ Repository Cleanup (COMPLETE)
- Deleted: demo_examples, doctest_*.py, internal TASK_*.md files
- Updated: docs/conf.py for src/ layout
- Added: MANIFEST.in, gitignore entries
- Kept: test_config.py (35 tests)

**Status:** 100% complete as of 2026-01-14

---

## 📝 Notes & Decisions

### Files Tracked in Repository
- ✅ **slacgs.pdf** - Thesis (linked in README)
- ✅ **learning_*.pdf** - Research paper (linked in README)
- ✅ **scripts/** - Publishing automation
- ✅ **docs/** - Sphinx source (to regenerate)
- ✅ **test/test_config.py** - Core test file
- ✅ **presentation.tex** - Beamer slides (NEW!)
- ✅ **presentation_sum.md** - Slide summary (NEW!)

### Already Ignored (.gitignore)
- output/ (sim results)
- .venv/, venv/ (virtual envs)
- docs/_build/ (generated docs)
- dist/, build/, *.egg-info/ (build artifacts)
- slacgs.toml (user config)

---

## 🎯 Current Priority

**Next 3 Tasks (in order):**
1. **POLISH-01 & POLISH-02**: Add rich progress bars + color-coded console output (CLI polish)
2. **TEST-01 through TEST-INFRA-01**: Comprehensive test suite
3. **DOC-01 & DOC-03**: Regenerate docs + update README
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
