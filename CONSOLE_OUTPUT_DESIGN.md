# SLACGS Console Output Design (Phase 1)

**Status**: Design finalized, ready for implementation  
**Date**: 2026-01-14  
**Module**: `src/slacgs/progress.py`

---

## Executive Summary

Replace old-style line-per-iteration console output with **rich live panels** that update in-place. Panel overwrites itself at each checkpoint, keeping console clean and readable (max 10 lines).

**Key insight**: Users don't need to see iteration 1, 2, 3, ... they need to see:
1. We started ✓
2. Progress on current cardinality (live % bar)
3. Loss estimates at checkpoints (every 10-20 iters)
4. We finished with summary ✓

---

## Architecture

### Module Structure
```
src/slacgs/
├── progress.py              # ProgressTracker class (standalone)
│   ├── ProgressTracker     # Main class
│   ├── SimulationMetrics   # Data class for tracking
│   └── console             # Rich Console instance
└── core/
    └── simulator.py         # Uses ProgressTracker
```

### Integration Points
1. **Simulator.__init__()**: Instantiate `ProgressTracker(verbose, debug)`
2. **Simulator.run()**: Call progress methods at key checkpoints
3. **cli.py**: Add `--debug` and `--quiet` flags

---

## Console Output Sequence

### 1️⃣ Initialization (On start)
**What it shows**:
- Green panel with model parameters
- Dimensionalities and cardinalities to process
- Output directory path

**Code trigger**: `progress.log_simulation_start(model, dims, N, output_path)`

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃  🚀 SLACGS Simulator Started             ┃
┃  Model params: [1, 4, 0.6]               ┃
┃  Dimensions: [2, 3]                      ┃
┃  Cardinalities: [2, 4, 8, 16, ..., 1024] ┃
┃  Mode: Normal (test_mode=False)          ┃
┃  Output: ~/slacgs/output/                ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

### 2️⃣ Cardinality Loop (Live Panel - Updates In-Place)
**What it shows**:
- Progress bar for current cardinality (n=2, 4, 8, etc.)
- Completion percentage and ETA
- Current iteration count
- Loss estimates per dimension (updated at each checkpoint)
- Convergence status per dimension

**Code trigger**: 
- `progress.start_cardinality_loop(N_values)` - Once at start of loop
- `progress.log_cardinality_start(n, dims, max_iter)` - Per cardinality
- `progress.update_checkpoint(iter, loss_sum, iter_N, dims)` - Every checkpoint
- `progress.log_cardinality_complete(n, elapsed)` - Per cardinality end

**Live Update Frequency**: Every checkpoint step (10-20 iters in test mode, 100-200 in full)

```
┏ Cardinality Progress ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
│ n=16       [████████░░░░░░░░░░░░░░░░░░░░] 40% ⏱ 2m 15s ETA │
│ 🔄 Training SVM (iteration 40/100)...                      │
│                                                            │
│ Loss Estimates @ Checkpoint:                              │
│   d=2: THEORY=0.4023 | TRAIN=0.3847 | TEST=0.3901 [🟢 ↘] │
│   d=3: THEORY=0.2156 | TRAIN=0.1945 | TEST=0.2034 [🟢 ↘] │
│                                                            │
│ Convergence: d=2 [improved] d=3 [improved]                │
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

**Loss Color Legend**:
- 🟢 Green: Loss improved significantly since last checkpoint
- 🟡 Yellow: Loss stagnating (within convergence threshold)
- 🔵 Blue: Neutral reference value

---

### 3️⃣ Simulation Complete (Final Summary)
**What it shows**:
- Total runtime
- Cardinalities processed
- Final loss values per dimension
- Output file locations (reports, data, logs)

**Code trigger**: `progress.log_simulation_complete(total_elapsed, cards_processed, dims)`

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ✅ Simulation Complete                              ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ 📊 Summary                                          ┃
┃    Total time:       12m 45s                        ┃
┃    Cardinalities:    10/10 processed                ┃
┃    Dimensions:       [2, 3]                         ┃
┃                                                     ┃
┃ 📈 Final Losses (Theoretical)                       ┃
┃    d=2: 0.3412 ✓                                    ┃
┃    d=3: 0.1654 ✓                                    ┃
┃                                                     ┃
┃ 📁 Output                                           ┃
┃    Reports: /home/pr/slacgs/output/reports/        ┃
┃    Data:    /home/pr/slacgs/output/data/           ┃
┃    Logs:    /home/pr/slacgs/output/logs/           ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

### 4️⃣ Error Handling (On Exception)
**What it shows**:
- Error type and message
- Context (which model params, iteration, etc.)
- Helpful hint for fixing

**Code trigger**: `progress.log_error(message, context)`

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ ❌ ERROR: Covariance matrix not positive ┃
┃           definite                       ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Model params: [1, -2, 0.5]               ┃
┃ Issue: At cardinality n=16               ┃
┃                                          ┃
┃ 💡 Hint: Ensure all σᵢ > 0 and |ρ| < 1  ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

---

## Verbosity Levels

Three modes controlled via CLI flags:

### Normal Mode (default)
```bash
slacgs run-simulation --params "[1,1,0.4]" --test-mode
```
Shows: Start message + checkpoint updates + end summary

### Debug Mode
```bash
slacgs run-simulation --params "[1,1,0.4]" --test-mode --debug
```
Shows: Normal + per-iteration detailed logs (useful for troubleshooting)

Example output with --debug:
```
┏ Cardinality Progress ... ┓
│ n=16 [progress bar]      │
│ 🔄 Training SVM...       │
│ Iter 1:   d=2 L=0.45     │  ← Only with --debug
│ Iter 2:   d=2 L=0.44     │  ← Old-style logging
│ ...                      │
│ [Checkpoint @ Iter 10]   │
│ Loss estimates...        │
└──────────────────────────┘
```

### Quiet Mode
```bash
slacgs run-experiment --scenarios 1 --test-mode --quiet
```
Shows: Only initialization message and final summary (good for batch/CI)

---

## Implementation Checklist

- [ ] **Simulator Integration**
  - [ ] Add `verbose` and `debug` parameters to `__init__()`
  - [ ] Instantiate `ProgressTracker(verbose, debug)` in `__init__()`
  - [ ] Call progress methods in `run()` at key points
  - [ ] Handle error cases with `progress.log_error()`

- [ ] **CLI Integration**
  - [ ] Add `--debug` flag to `run-simulation` command
  - [ ] Add `--quiet` flag to `run-simulation` command
  - [ ] Add `--debug` flag to `run-experiment` command
  - [ ] Add `--quiet` flag to `run-experiment` command
  - [ ] Pass these flags to Simulator

- [ ] **Testing**
  - [ ] Test normal mode: `slacgs run-experiment --scenarios 1 --test-mode`
  - [ ] Test debug mode: `slacgs run-experiment --scenarios 1 --test-mode --debug`
  - [ ] Test quiet mode: `slacgs run-experiment --scenarios 1 --test-mode --quiet`
  - [ ] Verify panel updates in-place (doesn't scroll)
  - [ ] Verify convergence detection shows correct colors

---

## Benefits

1. **Cleaner console**: No 100+ scrolling lines
2. **Real-time feedback**: See progress live, not after it's done
3. **Debugging support**: `--debug` flag for old-style iteration logs
4. **Batch-friendly**: `--quiet` mode for CI/automation
5. **Professional look**: Rich panels and colors

---

## Notes

- Uses `rich.progress.Progress` with live updates
- All panels auto-fit terminal width
- Colors are ANSI-safe (work in all terminals)
- Panel overwrites happen in-place (no new lines scrolled)
