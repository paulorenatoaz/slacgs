"""
Rich-based progress tracking for SLACGS simulations.

Provides:
- Beautiful console panels for simulation start, progress, and completion
- Live progress bar for cardinality iterations
- Color-coded loss estimates per dimensionality
- Status messages with timestamps
"""

from typing import Dict, List, Optional
import time
from dataclasses import dataclass, field

from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.columns import Columns
from rich.align import Align


console = Console()


@dataclass
class SimulationMetrics:
    """Track metrics during a single cardinality simulation."""
    n: int
    d: int
    loss_type: str
    loss_value: float = 0.0
    iterations: int = 0
    max_iterations: int = 0
    
    def __str__(self) -> str:
        pct = (self.iterations / self.max_iterations * 100) if self.max_iterations > 0 else 0
        return f"Loss: {self.loss_value:.4f} | {self.iterations}/{self.max_iterations} iters ({pct:.0f}%)"


class ProgressTracker:
    """
    Manages rich console progress display during simulation.
    
    Displays:
    - Initialization panel with model/simulator config
    - LIVE updating panel with progress bar and loss estimates (updates in-place)
    - Completion summary with timings and output locations
    """
    
    def __init__(self, verbose: bool = True, debug: bool = False):
        self.verbose = verbose
        self.debug = debug
        self.live_display = None
        self.current_cardinality = None
        self.current_iteration = 0
        self.max_iteration = 0
        self.start_time = None
        self.cardinality_start_time = None
        self.cardinalities_processed = 0
        self.total_cardinalities = 0
        self.current_dims = []
        self.current_losses: Dict[int, Dict[str, float]] = {}  # d -> loss_type -> current_value
        self.prev_losses: Dict[int, Dict[str, float]] = {}  # For convergence detection
    
    def log_simulation_init(self, model_params: List, dims: List[int], N: List[int], test_mode: bool = False, output_path: str = ""):
        """Display initialization panel with simulation configuration."""
        if not self.verbose:
            return
        
        mode_text = "[yellow]TEST MODE (10x faster)[/yellow]" if test_mode else "[cyan]NORMAL MODE[/cyan]"
        
        # Format cardinality range better
        if len(N) <= 5:
            card_text = f"{N}"
        else:
            card_text = f"{N[0]} → {N[-1]} (steps: {len(N)})"
        
        panel_content = (
            f"[bold green]🚀 SLACGS Simulation Started[/bold green]\n"
            f"\n[cyan]Model Configuration:[/cyan]\n"
            f"  Parameters: {model_params}\n"
            f"  Dimensionalities: {dims}\n"
            f"\n[cyan]Simulation Plan:[/cyan]\n"
            f"  Cardinalities: {card_text}\n"
            f"  Mode: {mode_text}\n"
            f"\n[cyan]Output:[/cyan]\n"
            f"  Path: {output_path}"
        )
        
        panel = Panel(panel_content, border_style="green", expand=False)
        console.print(panel)
    
    def start_cardinality_loop(self, n_values: List[int]):
        """Initialize live display for cardinality loop."""
        if not self.verbose:
            return
        
        self.start_time = time.time()
        self.total_cardinalities = len(n_values)
        self.cardinalities_processed = 0
        
        # Start live display
        try:
            self.live_display = Live(console=console, refresh_per_second=4, transient=False)
            self.live_display.start()
        except Exception as e:
            # Fallback if live display fails (e.g., in non-TTY)
            console.print(f"[yellow]Note: Live display unavailable, using fallback mode[/yellow]")
            self.live_display = None
    
    def _build_live_panel(self) -> Panel:
        """Build the live-updating progress panel."""
        if self.current_cardinality is None:
            return Panel("Initializing...", border_style="blue")
        
        # Calculate progress
        progress_pct = (self.current_iteration / self.max_iteration * 100) if self.max_iteration > 0 else 0
        elapsed = time.time() - self.cardinality_start_time if self.cardinality_start_time else 0
        
        # Estimate time remaining
        if self.current_iteration > 0:
            time_per_iter = elapsed / self.current_iteration
            remaining_iters = self.max_iteration - self.current_iteration
            eta_seconds = time_per_iter * remaining_iters
            eta_text = f"{int(eta_seconds // 60)}m {int(eta_seconds % 60)}s"
        else:
            eta_text = "calculating..."
        
        # Progress bar
        bar_width = 30
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Build content
        content = f"[bold blue]n={self.current_cardinality}[/bold blue]       [{bar}] {progress_pct:.0f}% ⏱ {eta_text} ETA\n"
        content += f"🔄 Training SVM (iteration {self.current_iteration}/{self.max_iteration})...\n"
        content += "\n"
        
        # Loss estimates table (if available)
        if self.current_losses:
            content += "[cyan]Loss Estimates @ Checkpoint:[/cyan]\n"
            for d in sorted(self.current_dims):
                if d in self.current_losses:
                    losses = self.current_losses[d]
                    theory = losses.get('THEORETICAL', 0.0)
                    train = losses.get('EMPIRICAL_TRAIN', 0.0)
                    test = losses.get('EMPIRICAL_TEST', 0.0)
                    
                    # Convergence indicator
                    if d in self.prev_losses and 'EMPIRICAL_TRAIN' in self.prev_losses[d]:
                        prev_train = self.prev_losses[d]['EMPIRICAL_TRAIN']
                        if abs(train - prev_train) < 0.001:
                            status = "[🟡 ≈]"
                        elif train < prev_train:
                            status = "[🟢 ↘]"
                        else:
                            status = "[🔵 ↗]"
                    else:
                        status = "[🔵 •]"
                    
                    content += f"  d={d}: THEORY={theory:.4f} | TRAIN={train:.4f} | TEST={test:.4f} {status}\n"
            content += "\n"
            
            # Convergence summary
            converged = sum(1 for d in self.current_dims if d in self.current_losses and 
                          d in self.prev_losses and 
                          abs(self.current_losses[d].get('EMPIRICAL_TRAIN', 0) - 
                              self.prev_losses[d].get('EMPIRICAL_TRAIN', 0)) < 0.001)
            content += f"Convergence: {converged}/{len(self.current_dims)} dims stabilized"
        
        return Panel(content, title="[bold cyan]Cardinality Progress[/bold cyan]", border_style="cyan", expand=False)
    
    def log_cardinality_start(self, n: int, dims: List[int], max_iter: int):
        """Start tracking a new cardinality (updates live panel)."""
        if not self.verbose:
            return
        
        self.current_cardinality = n
        self.current_dims = dims
        self.max_iteration = max_iter
        self.current_iteration = 0
        self.cardinality_start_time = time.time()
        self.current_losses = {}
        
        # Update live display or print message if no live display
        if self.live_display:
            self.live_display.update(self._build_live_panel())
        else:
            console.print(f"[blue]▶ Starting n={n}[/blue] | dims={dims} | max_iter={max_iter}")
    
    def update_checkpoint(self, iteration: int, loss_sum: Dict[int, Dict[str, float]], 
                         iter_count: Dict[int, Dict[str, int]], dims: List[int]):
        """Update the live panel with checkpoint data."""
        if not self.verbose:
            return
        
        self.current_iteration = iteration
        
        # Calculate average losses at this checkpoint
        self.prev_losses = self.current_losses.copy()
        self.current_losses = {}
        for d in dims:
            self.current_losses[d] = {}
            for loss_type in ['THEORETICAL', 'EMPIRICAL_TRAIN', 'EMPIRICAL_TEST']:
                if d in loss_sum and loss_type in loss_sum[d] and d in iter_count and loss_type in iter_count[d]:
                    count = iter_count[d][loss_type]
                    if count > 0:
                        self.current_losses[d][loss_type] = loss_sum[d][loss_type] / count
        
        # Update live display
        if self.live_display:
            self.live_display.update(self._build_live_panel())
    
    def log_cardinality_complete(self, n: int, elapsed_sec: float):
        """Mark cardinality as complete (updates live panel)."""
        if not self.verbose:
            return
        
        # Increment counter
        self.cardinalities_processed += 1
        
        # Show brief completion message in debug mode
        if self.debug:
            console.print(f"[green]✓[/green] n={n} complete in {elapsed_sec:.1f}s")
    
    def finish_cardinality_loop(self):
        """Close live display after cardinality loop."""
        if self.live_display:
            self.live_display.stop()
            self.live_display = None
    
    def update_cardinality_progress(self, n: int):
        """Mark a cardinality as finished (for progress tracking)."""
        # This is now handled by log_cardinality_complete
        pass
    
    def log_step_update(self, n: int, step: int, total_steps: int, dims: List[int], losses: Dict[int, Dict[str, float]]):
        """Update live loss table (called periodically during iteration)."""
        if not self.verbose or step % 10 != 0:  # Update every 10 steps to avoid spam
            return
        
        # Create a table showing current losses per dimension
        table = Table(title=f"Step {step}/{total_steps} - Cardinality n={n}", show_header=True, header_style="bold cyan")
        table.add_column("Dimension", style="cyan", width=10)
        table.add_column("EMPIRICAL_TRAIN", justify="right", width=14)
        table.add_column("EMPIRICAL_TEST", justify="right", width=14)
        table.add_column("THEORETICAL", justify="right", width=14)
        
        for d in dims:
            if d not in losses:
                continue
            train_loss = losses[d].get('EMPIRICAL_TRAIN', 0.0)
            test_loss = losses[d].get('EMPIRICAL_TEST', 0.0)
            theo_loss = losses[d].get('THEORETICAL', 0.0)
            
            table.add_row(
                str(d),
                f"{train_loss:.4f}",
                f"{test_loss:.4f}",
                f"{theo_loss:.4f}"
            )
        
        console.print(table)
    
    def log_simulation_complete(self, elapsed_sec: float, total_cardinalities: int, total_dimensions: int, reports_dir: str = "", data_dir: str = ""):
        """Log final summary after simulation completes with beautiful panel."""
        if not self.verbose:
            return
        
        hours = elapsed_sec / 3600
        minutes = (elapsed_sec % 3600) / 60
        secs = elapsed_sec % 60
        
        # Format time nicely
        if hours >= 1:
            time_str = f"{hours:.1f}h {minutes:.0f}m"
        elif minutes >= 1:
            time_str = f"{minutes:.0f}m {secs:.0f}s"
        else:
            time_str = f"{secs:.1f}s"
        
        panel_content = (
            f"[bold green]✅ Simulation Complete[/bold green]\n"
            f"\n[cyan]Execution Summary:[/cyan]\n"
            f"  Total time: [yellow]{time_str}[/yellow]\n"
            f"  Cardinalities processed: [yellow]{total_cardinalities}[/yellow]\n"
            f"  Dimensions analyzed: [yellow]{total_dimensions}[/yellow]\n"
        )
        
        # Add output paths if provided
        if reports_dir or data_dir:
            panel_content += f"\n[cyan]Output Files:[/cyan]\n"
            if reports_dir:
                panel_content += f"  📊 Reports: [yellow]{reports_dir}[/yellow]\n"
            if data_dir:
                panel_content += f"  📁 Data: [yellow]{data_dir}[/yellow]\n"
        
        panel = Panel(panel_content, border_style="green", expand=False)
        console.print(panel)
    
    def log_error(self, message: str):
        """Log an error message."""
        msg = Text(f"✗ ERROR: {message}", style="bold red")
        console.print(msg)
    
    def log_warning(self, message: str):
        """Log a warning message."""
        msg = Text(f"⚠ WARNING: {message}", style="bold yellow")
        console.print(msg)
    
    def log_info(self, message: str):
        """Log an info message."""
        msg = Text(f"ℹ {message}", style="bold blue")
        console.print(msg)


# Global progress tracker instance
_tracker: Optional[ProgressTracker] = None


def get_progress_tracker(verbose: bool = True) -> ProgressTracker:
    """Get or create the global progress tracker."""
    global _tracker
    if _tracker is None:
        _tracker = ProgressTracker(verbose=verbose)
    return _tracker


def reset_progress_tracker():
    """Reset the global progress tracker."""
    global _tracker
    _tracker = None
