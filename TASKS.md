# 🚀 CoInfoSim Publish Checklist (0.2.0)

Short, actionable steps to publish v0.2.0 while keeping the full history below.

## Version Status
- Current code version: 0.2.0 (setup.py and src/coinfosim/__init__.py)
- PyPI latest: check after upload

## Do Now
- [ ] Regenerate Sphinx docs
  - `cd docs && make clean && make html`
- [ ] Review README (links to PDFs, CLI quickstart)
- [ ] Build & validate
  - `pip install -U build twine`
  - `python -m build`
  - `twine check dist/*`
- [ ] Test install (wheel)
  - `python -m venv .venv-publish && source .venv-publish/bin/activate`
  - `pip install dist/coinfosim-*.whl`
  - `coinfosim --help`
- [ ] TestPyPI (optional)
  - `twine upload --repository testpypi dist/*`
  - `pip install --index-url https://test.pypi.org/simple/ coinfosim`
- [ ] Publish to PyPI
  - `twine upload dist/*`
- [ ] Publish reports to Pages
  - `bash scripts/publish_output_to_pages.sh`
- [ ] Tag release v0.2.0
  - `git tag -a v0.2.0 -m "Release v0.2.0" && git push origin v0.2.0`

---

# 📋 CoInfoSim Development Tasks

**Last Updated:** 2026-01-14  
**Current Phase:** Repository Cleanup & Publication Preparation

---

## 🎯 Quick Status

| Category | Complete | Total | Progress |
|----------|----------|-------|----------|
| ✅ Core Architecture | 17 | 17 | 100% |
| 🚧 Cleanup & Publication | 5 | 9 | 56% |
| 📦 Testing & Quality | 0 | 5 | 0% |
| 🚀 Release Preparation | 0 | 3 | 0% |
| 🌟 Future Enhancements | 0 | 3 | 0% |
| **TOTAL** | **22** | **37** | **59%** |

---

## ✅ Phase 1: Core Architecture (COMPLETE!)

### Package Structure & Imports
- [x] **001** - GitHub Pages setup for reports publication
- [x] **002** - Publishing workflow (scripts/publish_output_to_pages.sh)
- [x] **010** - Migrate to src/ layout
- [x] **011** - Create subpackages (core/, reporting/, publish/, legacy/)
- [x] **012** - Convert to absolute imports
- [x] **020** - Fix duplicate `__all__` in `__init__.py`
- [x] **021** - Remove wildcard imports
- [x] **022** - Add `__all__` to all subpackage `__init__.py` files

### Configuration & Logging Infrastructure
- [x] **023** - Update setup.py dependencies (CLI tools, extras_require)
- [x] **024** - Create config.py (TOML-based, 544 lines, 35 tests)
  - Priority: CLI > Env > ./coinfosim.toml > ~/.config/coinfosim/ > defaults
  - Config is OPTIONAL (sane defaults work out of the box)
- [x] **025** - Create logging_config.py (rotating logs, Rich output)

### CLI Implementation
- [x] **026** - Create cli.py with typer framework (570+ lines)
- [x] **027** - Implement all CLI commands (run-simulation, run-experiment, make-report, publish, config)
- [x] **028** - ~~Output directory management~~ *Merged into 024/027*

### Data Architecture Refactor
- [x] **030** - Create ReportData dataclass (pure data object, JSON serializable)
- [x] **031** - Refactor Report to use ReportData (removed all 112 `self.sim` references)
- [x] **032** - Update Simulator to produce ReportData (eliminated circular dependency)

**🎉 Result:** Modern CLI-first package with clean architecture, no circular deps, full TOML config support

---

## 🚧 Phase 2: Cleanup & Publication Prep (IN PROGRESS)

### Repository Cleanup ✅ COMPLETED 2026-01-14
- [x] **CLEANUP-01** - Clean test/ folder
  - ✅ Deleted: demo_examples.py, demo_examples_test.py, doctest_*.py
  - ✅ Kept: test_config.py (443 lines, 35 tests)
- [x] **CLEANUP-02** - Fix docs/source/conf.py paths for src/ layout
  - ✅ Updated sys.path to `../../src` (was `..\\coinfosim`)
- [x] **CLEANUP-03** - Create MANIFEST.in for PyPI distribution control
- [x] **CLEANUP-04** - Update .gitignore
  - ✅ Added: docs/_build/, coinfosim.toml, task tracking patterns
- [x] **CLEANUP-05** - Remove internal task tracking files
  - ✅ Deleted: TASK_025_COMPLETE.md, TASKS_026_027_COMPLETE.md, etc.
  - ✅ Kept: TASKS.md (this file - active development tracker)

### Documentation & Publication
- [ ] **DOC-01** - Regenerate Sphinx documentation
  - [ ] Verify docs/source/conf.py configuration
  - [ ] Run: `cd docs && make clean && make html`
  - [ ] Test: Open docs/_build/html/index.html
  - [ ] **DECISION NEEDED:** Publish docs to GitHub Pages? Or just include in repo?
    - Option A: Docs in repo only (users build locally or view on GitHub)
    - Option B: Publish to gh-pages branch (separate from reports-pages)
    - Option C: Publish to Read the Docs (external hosting)
  
- [ ] **DOC-02** - Update README.md for publication
  - [ ] Add installation: `pip install coinfosim`
  - [ ] Add quick start with CLI examples
  - [ ] Add badges (PyPI version, Python versions, license, CI status)
  - [ ] Ensure coinfosim.pdf and learning_with_few_features_and_samples.pdf links work
  - [ ] Add "Citation" section pointing to CITATION.cff (see DOC-03)

- [ ] **DOC-03** - Create scientific citation metadata
  - [ ] Create CITATION.cff for GitHub citation feature
  - [ ] Add citation instructions to README
  - [ ] Include thesis and paper references

### PyPI Preparation
- [ ] **PYPI-01** - Test PyPI build locally
  - [ ] Install build tools: `pip install build twine`
  - [ ] Build: `python -m build`
  - [ ] Check: `twine check dist/*`
  - [ ] Test install in clean venv: `pip install dist/coinfosim-*.whl`
  - [ ] Verify: `coinfosim --version`, `coinfosim --help`

- [ ] **PYPI-02** - Test publish to TestPyPI
  - [ ] Create TestPyPI account: https://test.pypi.org/account/register/
  - [ ] Configure ~/.pypirc with TestPyPI credentials
  - [ ] Upload: `twine upload --repository testpypi dist/*`
  - [ ] Test install: `pip install --index-url https://test.pypi.org/simple/ coinfosim`
  - [ ] Verify functionality

---

## 📦 Phase 3: Testing & Quality

- [ ] **TEST-01** - Expand test suite (target: >80% coverage)
  - [ ] Create test_model.py (unit tests for Model class)
  - [ ] Create test_simulator.py (unit tests for Simulator)
  - [ ] Create test_report.py (unit tests for Report)
  - [ ] Expand test_config.py (already has 35 tests)
  - [ ] Create test_cli.py (use typer.testing.CliRunner)
  - [ ] Create test_integration.py (end-to-end workflows)
  - [ ] Add pytest.ini with coverage settings
  - [ ] Run: `pytest --cov=coinfosim --cov-report=html --cov-report=term`

- [ ] **QUALITY-01** - Set up code quality tools
  - [ ] Create pyproject.toml with tool configs
  - [ ] Configure black (code formatting)
  - [ ] Configure ruff (linting, import sorting)
  - [ ] Configure mypy (type checking)
  - [ ] Create Makefile with: format, lint, typecheck, test, coverage
  - [ ] Run initial cleanup: `make format && make lint`

- [ ] **QUALITY-02** - Remove dead code
  - [ ] Find unused code with vulture
  - [ ] Remove commented-out code blocks
  - [ ] Remove unused imports (use ruff or autoflake)
  - [ ] Clean up TODO/FIXME comments

- [ ] **QUALITY-03** - Deprecate legacy code
  - [ ] Add module-level deprecation warning to demo.py
  - [ ] Update copilot-instructions.md (remove demo_scripts references)
  - [ ] Verify legacy/ modules have deprecation warnings
  - [ ] Update README to show CLI-only examples

- [ ] **CI-01** - Set up GitHub Actions CI/CD
  - [ ] Create .github/workflows/test.yml
    - Test matrix: Python 3.8, 3.9, 3.10, 3.11, 3.12
    - OS matrix: ubuntu-latest, macos-latest, windows-latest
    - Steps: checkout, setup-python, install deps, run tests, upload coverage
  - [ ] Create .github/workflows/publish.yml (triggered on release tags)
  - [ ] Add status badges to README
  - [ ] Optional: Set up codecov.io for coverage reporting

---

## 🚀 Phase 4: Release v1.0.0

- [ ] **RELEASE-01** - Pre-release checklist
  - [ ] All tests passing on all platforms
  - [ ] Documentation complete and builds without errors
  - [ ] CHANGELOG.md created with all changes since v0.1.9
  - [ ] Version bumped to 1.0.0 in:
    - [ ] src/coinfosim/__init__.py (__version__)
    - [ ] setup.py (version=)
  - [ ] README.md has correct PyPI install instructions
  - [ ] All deprecation warnings in place

- [ ] **RELEASE-02** - Publish to PyPI
  - [ ] Clean previous builds: `rm -rf dist/ build/ *.egg-info`
  - [ ] Build: `python -m build`
  - [ ] Final check: `twine check dist/*`
  - [ ] Upload: `twine upload dist/*`
  - [ ] Verify on PyPI: https://pypi.org/project/coinfosim/
  - [ ] Test install: `pip install coinfosim` (from fresh venv)
  - [ ] Test CLI: `coinfosim --version`, `coinfosim run-simulation --help`

- [ ] **RELEASE-03** - Create GitHub Release
  - [ ] Tag version: `git tag -a v1.0.0 -m "Release v1.0.0"`
  - [ ] Push tag: `git push origin v1.0.0`
  - [ ] Create GitHub Release with:
    - [ ] Release notes from CHANGELOG
    - [ ] Attach dist/ files (wheel, tarball)
    - [ ] Installation instructions
    - [ ] Citation information
  - [ ] Announce release (Twitter/LinkedIn/Reddit if applicable)
  - [ ] Update repo description and topics on GitHub
  - [ ] Monitor PyPI downloads and GitHub issues

---

## 🌟 Phase 5: Future Enhancements (Optional)

- [ ] **ENHANCE-01** - Community & Contribution Setup
  - [ ] Create CONTRIBUTING.md with development setup instructions
  - [ ] Create CODE_OF_CONDUCT.md
  - [ ] Create GitHub issue templates (bug, feature request, question)
  - [ ] Create GitHub PR template with checklist
  - [ ] Set up GitHub Discussions
  - [ ] Add "good first issue" labels to easy tasks

- [ ] **ENHANCE-02** - Enhanced Reports & Visualizations
  - [ ] Add interactive plots with plotly (optional dependency)
  - [ ] Add PDF export functionality (reportlab/weasyprint)
  - [ ] Create Jinja2 templates for customizable reports
  - [ ] Add CLI options: --theme, --export-pdf, --template

- [ ] **ENHANCE-03** - Advanced Configuration
  - [ ] Evaluate adding simulation defaults to config.toml
  - [ ] Examples: default test_samples_amt, default N ranges
  - [ ] ⚠️ **Important:** Keep seeds as per-simulation params (reproducibility)
  - [ ] Add after v1.0 release based on user feedback

---

## 📝 Notes & Decisions

### Files Kept in Repository
- ✅ **coinfosim.pdf** - Undergraduate thesis (linked in README)
- ✅ **learning_with_few_features_and_samples.pdf** - Research paper (linked in README)
- ✅ **scripts/** - Publishing automation (publish_output_to_pages.sh, setup_pages.sh)
- ✅ **docs/** - Sphinx documentation source (needs regeneration)
- ✅ **test/test_config.py** - Only test file (others deleted)

### Files Removed
- ❌ TCC/ folder - Deleted (entire LaTeX thesis project - unrelated to package)
- ❌ test/demo_examples*.py - Legacy Google Sheets demo code
- ❌ test/doctest_*.py - Redundant (doctests run via pytest)
- ❌ TASK_*.md, TASKS_*.md, REFACTOR_PLAN.md - Internal development notes

### Already Ignored by .gitignore
- output/ (simulation results)
- .venv/, venv/ (virtual environments)
- __pycache__/ (bytecode cache)
- *.egg-info/ (build artifacts)
- docs/_build/ (generated docs)
- coinfosim.toml (user-specific config)

---

## 🎯 Current Priority: Documentation & PyPI Testing

**Next 3 Tasks:**
1. DOC-01: Regenerate Sphinx docs & decide on hosting
2. PYPI-01: Test local build and wheel installation
3. DOC-02: Update README with modern badges and PyPI install instructions

