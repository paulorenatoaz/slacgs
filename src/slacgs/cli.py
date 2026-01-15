"""
SLACGS Command Line Interface.

Beautiful CLI for running simulations, generating reports, and managing experiments.

Usage:
    slacgs --help
    slacgs run-simulation --params "[1,4,0.6]"
    slacgs run-experiment --scenarios 1,2,3
    slacgs make-report --scenario 1
    slacgs publish
    slacgs config show
"""

import sys
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import print as rprint

from slacgs import __version__
from slacgs.config import load_config, init_project_config, validate_config, ConfigError, get_output_dir, get_reports_dir, get_data_dir
from slacgs.logging_config import setup_logging_from_config, get_logger
import re

# Create Typer app
app = typer.Typer(
    name="slacgs",
    help="SLACGS - Simulator for Loss Analysis of Classifiers on Gaussian Samples",
    add_completion=False,
    rich_markup_mode="rich",
)

# Rich console for beautiful output
console = Console()

# Global logger (set up after callback)
logger = None


def get_next_scenario_id(reports_dir: Path, test_mode: bool) -> int:
    """Find the next available scenario ID by checking existing scenario reports."""
    if not reports_dir.exists():
        return 1
    
    test_suffix = '[test]' if test_mode else ''
    pattern = re.compile(rf'scenario_(\d+)_report{re.escape(test_suffix)}\.html')
    max_id = 0
    
    for file in reports_dir.glob(f'scenario_*_report*.html'):
        match = pattern.match(file.name)
        if match:
            scenario_id = int(match.group(1))
            max_id = max(max_id, scenario_id)
    
    return max_id + 1


@app.callback()
def main_callback(
    ctx: typer.Context,
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL",
    ),
    log_file: Optional[str] = typer.Option(
        None,
        "--log-file",
        help="Log file path (default: from config or no file logging)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress console output (logs only to file)",
    ),
    no_color: bool = typer.Option(
        False,
        "--no-color",
        help="Disable colored output",
    ),
    config_file: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config file (default: ./slacgs.toml or ~/.config/slacgs/config.toml)",
    ),
):
    """
    Global options for SLACGS CLI.
    
    Configure logging, output, and load configuration before running any command.
    """
    global logger
    
    # Load configuration
    try:
        if config_file:
            config = load_config(config_path=Path(config_file))
        else:
            config = load_config()
    except ConfigError as e:
        console.print(f"[red]✗[/red] Configuration error: {e}", style="bold red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to load config: {e}", style="bold red")
        raise typer.Exit(1)
    
    # Setup logging with CLI overrides
    cli_overrides = {
        "log_level": log_level,
        "log_file": log_file,
        "quiet": quiet,
        "no_color": no_color,
    }
    
    try:
        setup_logging_from_config(config, cli_overrides)
        logger = get_logger("slacgs.cli")
        
        # Phase 2: Log session metadata
        from slacgs.logging_config import log_session_start
        from slacgs import __version__
        log_session_start(
            command=ctx.invoked_subcommand or "help",
            version=__version__,
            output_dir=str(config["paths"]["output_dir"]),
            config_source=config.get("_source", "defaults"),
        )
        
        logger.debug(f"CLI started with command: {ctx.invoked_subcommand}")
        logger.debug(f"Config loaded from: {config.get('_source', 'defaults')}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to setup logging: {e}", style="bold red")
        raise typer.Exit(1)
    
    # Store config in context for subcommands
    ctx.obj = {"config": config, "quiet": quiet}


@app.command()
def version():
    """Show version information."""
    panel = Panel(
        f"[bold]SLACGS[/bold] version [cyan]{__version__}[/cyan]\n\n"
        f"[dim]Simulator for Loss Analysis of Classifiers on Gaussian Samples[/dim]\n"
        f"[dim]https://github.com/paulorenatoaz/slacgs[/dim]",
        title="SLACGS",
        border_style="cyan",
    )
    console.print(panel)


@app.command(name="run-simulation")
def run_simulation(
    ctx: typer.Context,
    params: str = typer.Option(
        ...,
        "--params",
        "-p",
        help="Model parameters as JSON array, e.g., '[1,4,0.6]' for 2D or '[1,1,2,0,0,0]' for 3D",
    ),
    test_mode: bool = typer.Option(
        False,
        "--test-mode",
        "-t",
        help="Run in test mode (10x faster, reduced precision)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show per-iteration debug logs (verbose output)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress console output during simulation (summary only)",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (default: from config or ./output)",
    ),
):
    """
    Run a single simulation with specified parameters.
    
    Examples:
        slacgs run-simulation --params "[1,4,0.6]"
        slacgs run-simulation --params "[1,1,2,0,0,0]" --test-mode
        slacgs run-simulation --params "[1,1,0.4]" --test-mode --debug
        slacgs run-simulation --params "[1,1,0.4]" --test-mode --quiet
    """
    logger = get_logger("slacgs.cli.run_simulation")
    config = ctx.obj["config"]
    
    # Determine verbose mode (not quiet by default)
    verbose_mode = not quiet
    
    # Override output_dir if specified
    if output_dir:
        config["paths"]["output_dir"] = output_dir
        # Reinitialize report paths with new config
        from slacgs.utils import init_report_service_conf, report_service_conf
        report_service_conf.update(init_report_service_conf(config))
        
        # Reconfigure logging with new output directory
        from slacgs.logging_config import setup_logging_from_config
        setup_logging_from_config(config, force_reconfigure=True)
    
    if verbose_mode:
        console.print(f"\n[bold cyan]Running simulation[/bold cyan]")
        console.print(f"Parameters: [yellow]{params}[/yellow]")
        console.print(f"Test mode: [yellow]{test_mode}[/yellow]")
        if debug:
            console.print(f"Debug mode: [yellow]ON[/yellow]")
        console.print(f"Output: [cyan]{config['paths']['output_dir']}[/cyan]\n")
    
    try:
        # Parse parameters
        import json
        param_list = json.loads(params)
        
        logger.info(f"Starting simulation with params: {param_list}")
        
        # Import here to avoid loading heavy dependencies at CLI startup
        from slacgs import Model, Simulator
        
        # Create model
        if verbose_mode:
            console.print("Creating model...")
        model = Model(param_list)
        logger.debug(f"Model created: {model}")
        
        # Create simulator with verbose and debug flags
        if verbose_mode:
            console.print("Creating simulator...")
        simulator = Simulator(model, test_mode=test_mode, verbose=verbose_mode, debug=debug)
        
        # Run simulation
        if verbose_mode:
            console.print("Running simulation (this may take a while)...")
        with console.status("[bold green]Computing...[/bold green]"):
            simulator.run()
        
        logger.info("Simulation completed successfully")
        
        # Save results
        if verbose_mode:
            console.print("Saving results...")
        
        # Compute N* matrices between all dimensions (needed for report tables)
        simulator.report.print_N_star_matrix_between_all_dims()
        
        simulator.report.save_graphs_png_images_files()
        simulator.report.create_report_tables()
        simulator.report.write_to_json()
        simulator.report.create_html_report()
        
        if verbose_mode:
            console.print("\n[bold green]✓[/bold green] All results saved successfully!\n")
        
    except json.JSONDecodeError as e:
        console.print(f"[red]✗[/red] Invalid parameters format: {e}", style="bold red")
        logger.error(f"Parameter parsing failed: {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Simulation failed: {e}", style="bold red")
        logger.error(f"Simulation error: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command(name="run-experiment")
def run_experiment(
    ctx: typer.Context,
    scenarios: Optional[str] = typer.Option(
        None,
        "--scenarios",
        "-s",
        help="Comma-separated scenario numbers (e.g., '1,2,3'). Runs all predefined scenarios if not specified.",
    ),
    custom_params: Optional[str] = typer.Option(
        None,
        "--custom-params",
        "-c",
        help="Custom parameter sets as JSON array (e.g., '[[1,4,0.6], [1,2,0.5]]')",
    ),
    params_file: Optional[str] = typer.Option(
        None,
        "--params-file",
        "-f",
        help="Load parameter sets from JSON file",
    ),
    test_mode: bool = typer.Option(
        False,
        "--test-mode",
        "-t",
        help="Run in test mode (10x faster, reduced precision)",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Show per-iteration debug logs (verbose output)",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Suppress console output during simulations (summary only)",
    ),
    skip_existing: bool = typer.Option(
        True,
        "--skip-existing/--no-skip-existing",
        help="Skip simulations already in simulation_reports.json",
    ),
    output_dir: Optional[str] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (default: from config or user home directory)",
    ),
    tag: Optional[str] = typer.Option(
        None,
        "--tag",
        help="Organize this experiment under ~/slacgs/experiments/{tag}/ (ignored if --output-dir is set)",
    ),
):
    """
    Run multiple simulations from predefined or custom scenarios.
    
    Examples:
        # Predefined scenarios
        slacgs run-experiment
        slacgs run-experiment --scenarios 1,2,3
        
        # Custom inline parameters
        slacgs run-experiment --custom-params "[[1,4,0.6], [1,2,0.5]]"
        
        # Load from file
        slacgs run-experiment --params-file my_scenario.json
        
        # Organize with tag (creates ~/slacgs/experiments/{tag}/)
        slacgs run-experiment --params-file exp.json --tag paper1_v1
        slacgs run-experiment --scenarios 1,2,3 --tag demo
        
        # Full control with output directory (overrides --tag)
        slacgs run-experiment --params-file exp.json --output-dir ./results/batch_001
        
        # Test mode
        slacgs run-experiment --scenarios 1 --test-mode
    """
    logger = get_logger("slacgs.cli.run_experiment")
    config = ctx.obj["config"]
    
    # Handle tag for experiment organization (before output_dir override)
    if tag and not output_dir:
        # Validate tag (alphanumeric, underscore, dash only)
        if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
            console.print(
                f"[red]✗[/red] Invalid tag '{tag}': must contain only letters, numbers, underscore, or dash",
                style="bold red"
            )
            raise typer.Exit(1)
        
        # Set output to experiments/{tag}/
        output_dir = str(Path.home() / "slacgs" / "experiments" / tag)
        console.print(f"[dim]→ Using tagged experiment folder: {output_dir}[/dim]")
    
    # Override output_dir if specified (from --tag or --output-dir)
    if output_dir:
        config["paths"]["output_dir"] = output_dir
        # Reinitialize report paths with new config
        from slacgs.utils import init_report_service_conf, report_service_conf
        report_service_conf.update(init_report_service_conf(config))
        
        # Reconfigure logging with new output directory
        from slacgs.logging_config import setup_logging_from_config
        setup_logging_from_config(config, force_reconfigure=True)
    
    console.print(f"\n[bold cyan]Running experiment[/bold cyan]")
    console.print(f"Output directory: [cyan]{config['paths']['output_dir']}[/cyan]\n")
    
    try:
        # Import heavy dependencies
        from slacgs import Model, Simulator
        from slacgs.utils import is_param_in_simulation_reports
        import json
        
        # Validate mutually exclusive options
        mode_count = sum([scenarios is not None, custom_params is not None, params_file is not None])
        if mode_count > 1:
            console.print("[red]ERROR:[/red] Cannot combine --scenarios, --custom-params, and --params-file", style="bold red")
            console.print("Choose one: predefined scenarios, custom inline params, or params from file")
            raise typer.Exit(1)
        
        # Determine experiment mode and load parameter sets
        all_param_sets = []
        mode_description = ""
        scenario_list = []  # Track scenarios for report generation
        
        if custom_params:
            # Mode: Custom inline parameters
            try:
                loaded_data = json.loads(custom_params)
                if not isinstance(loaded_data, list):
                    raise ValueError("custom-params must be a JSON array")
                
                # Detect if this is multiple scenarios (nested array) or single scenario (flat array)
                if loaded_data and isinstance(loaded_data[0], list):
                    # Check if this is nested (array of scenarios)
                    if loaded_data[0] and isinstance(loaded_data[0][0], list):
                        # Nested: [[[1,3.1,0],[1,3.2,0]], [[1,2,0.2],[1,2,0.3]]]
                        # Each sub-array is a scenario
                        console.print(f"Running [yellow]{len(loaded_data)}[/yellow] custom scenarios\n")
                        for idx, scenario in enumerate(loaded_data, 1):
                            console.print(f"  Scenario {idx}: {len(scenario)} parameter sets")
                            scenario_list.append((None, scenario))
                            all_param_sets.extend(scenario)
                        mode_description = "custom scenarios"
                    else:
                        # Flat: [[1,3.1,0],[1,3.2,0]]
                        # Single scenario
                        all_param_sets = loaded_data
                        scenario_list = [(None, all_param_sets)]
                        console.print(f"Running [yellow]{len(all_param_sets)}[/yellow] custom simulations\n")
                        mode_description = "custom inline parameters"
                else:
                    # Empty or invalid
                    all_param_sets = loaded_data
                    scenario_list = [(None, all_param_sets)]
                    console.print(f"Running [yellow]{len(all_param_sets)}[/yellow] custom simulations\n")
                    mode_description = "custom inline parameters"
                    
            except json.JSONDecodeError as e:
                console.print(f"[red]ERROR:[/red] Invalid JSON in --custom-params: {e}", style="bold red")
                raise typer.Exit(1)
        
        elif params_file:
            # Mode: Load from file
            try:
                file_path = Path(params_file)
                if not file_path.exists():
                    console.print(f"[red]ERROR:[/red] File not found: {params_file}", style="bold red")
                    raise typer.Exit(1)
                
                with open(file_path, 'r') as f:
                    loaded_data = json.load(f)
                
                if not isinstance(loaded_data, list):
                    raise ValueError("params file must contain a JSON array")
                
                # Detect if this is multiple scenarios (nested array) or single scenario (flat array)
                # Check if first element is a list and contains numeric values (params)
                if loaded_data and isinstance(loaded_data[0], list):
                    # Check if this is nested (array of scenarios)
                    if loaded_data[0] and isinstance(loaded_data[0][0], list):
                        # Nested: [[params1, params2], [params3, params4]]
                        # Each sub-array is a scenario
                        console.print(f"Loaded [yellow]{len(loaded_data)}[/yellow] scenarios from [cyan]{params_file}[/cyan]")
                        for idx, scenario in enumerate(loaded_data, 1):
                            console.print(f"  Scenario {idx}: {len(scenario)} parameter sets")
                            scenario_list.append((None, scenario))
                            all_param_sets.extend(scenario)
                        mode_description = f"scenarios from {params_file}"
                    else:
                        # Flat: [params1, params2, params3]
                        # Single scenario
                        all_param_sets = loaded_data
                        scenario_list = [(None, all_param_sets)]
                        console.print(f"Loaded [yellow]{len(all_param_sets)}[/yellow] simulations from [cyan]{params_file}[/cyan]\n")
                        mode_description = f"parameters from {params_file}"
                else:
                    # Empty or invalid
                    all_param_sets = loaded_data
                    scenario_list = [(None, all_param_sets)]
                    console.print(f"Loaded [yellow]{len(all_param_sets)}[/yellow] simulations from [cyan]{params_file}[/cyan]\n")
                    mode_description = f"parameters from {params_file}"
                
            except json.JSONDecodeError as e:
                console.print(f"[red]ERROR:[/red] Invalid JSON in {params_file}: {e}", style="bold red")
                raise typer.Exit(1)
            except Exception as e:
                console.print(f"[red]ERROR:[/red] Failed to load {params_file}: {e}", style="bold red")
                raise typer.Exit(1)
        
        else:
            # Mode: Predefined scenarios (default)
            from slacgs.demo import SCENARIOS
            
            scenario_filter = None
            if scenarios:
                scenario_filter = [int(s.strip()) - 1 for s in scenarios.split(",")]
                console.print(f"Running predefined scenarios: [yellow]{scenarios}[/yellow]\n")
            else:
                console.print(f"Running [yellow]all {len(SCENARIOS)}[/yellow] predefined scenarios\n")
            
            # Build scenario list for reports and flatten params
            for idx, scenario in enumerate(SCENARIOS):
                if scenario_filter and idx not in scenario_filter:
                    continue
                scenario_list.append((idx + 1, scenario))
                all_param_sets.extend(scenario)
            
            mode_description = "predefined scenarios"
        
        # Count total simulations
        total_simulations = len(all_param_sets)
        console.print(f"Total simulations to process: [yellow]{total_simulations}[/yellow]\n")
        
        # Run simulations
        completed = 0
        skipped = 0
        
        for param in all_param_sets:
            # Check if already exists
            if skip_existing and is_param_in_simulation_reports(param, test_mode=test_mode):
                console.print(f"  [dim]SKIP: {param} (already exists)[/dim]")
                skipped += 1
                continue
            
            console.print(f"  [cyan]>>>[/cyan] Running {param}...")
            logger.info(f"Simulation {completed + 1}/{total_simulations}: {param}")
            
            # Run simulation
            try:
                model = Model(param)
                # In experiments, use full N range even in test mode for scenario report compatibility
                # Determine verbose mode (not quiet by default)
                verbose_mode = not quiet
                simulator = Simulator(model, test_mode=test_mode, full_n_range=True, verbose=verbose_mode, debug=debug)
                
                with console.status(f"[bold green]Computing {param}...[/bold green]"):
                    simulator.run()
                
                # Save results
                # Compute N* matrices between all dimensions (needed for report tables)
                simulator.report.print_N_star_matrix_between_all_dims()
                
                simulator.report.save_graphs_png_images_files()
                simulator.report.create_report_tables()
                simulator.report.write_to_json()
                simulator.report.create_html_report()
                
                console.print(f"  [green]OK:[/green] Completed {param}")
                completed += 1
                
            except Exception as e:
                console.print(f"  [red]ERROR:[/red] Failed {param}: {e}")
                logger.error(f"Simulation failed for {param}: {e}", exc_info=True)
        
        # Summary
        console.print(f"\n[bold green]SUCCESS:[/bold green] Experiment complete!")
        console.print(f"  Mode: [cyan]{mode_description}[/cyan]")
        console.print(f"  Completed: [green]{completed}[/green]")
        console.print(f"  Skipped: [yellow]{skipped}[/yellow]")
        console.print(f"  Total: {completed + skipped}/{total_simulations}\n")
        
        # DEBUG
        console.print(f"[yellow]DEBUG: completed={completed}, len(scenario_list)={len(scenario_list)}[/yellow]")
        if scenario_list:
            console.print(f"[yellow]DEBUG: scenario_list={scenario_list}[/yellow]")
        
        # Generate scenario reports
        if completed > 0 and scenario_list:
            console.print("[bold cyan]Generating scenario reports...[/bold cyan]\n")
            from slacgs.reporting.report_utils import create_scenario_report
            
            # Get reports directory
            reports_dir = Path(get_reports_dir(config))
            
            # Assign numeric IDs to scenarios that don't have one yet
            next_id = get_next_scenario_id(reports_dir, test_mode)
            updated_scenario_list = []
            for scenario_id, scenario_params in scenario_list:
                if scenario_id is None:
                    scenario_id = next_id
                    next_id += 1
                updated_scenario_list.append((scenario_id, scenario_params))
            
            for scenario_id, scenario_params in updated_scenario_list:
                try:
                    console.print(f"  Creating report for scenario: [yellow]{scenario_id}[/yellow]...")
                    create_scenario_report(scenario_params, scenario_id, test_mode=test_mode)
                    console.print(f"  [green]OK:[/green] Scenario {scenario_id} report created")
                except Exception as e:
                    console.print(f"  [red]ERROR:[/red] Failed to create scenario {scenario_id} report: {e}")
                    logger.error(f"Scenario report creation failed: {e}", exc_info=True)
            
            console.print(f"\n[bold green]SUCCESS:[/bold green] All scenario reports generated!\n")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Experiment failed: {e}", style="bold red")
        logger.error(f"Experiment error: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command(name="make-report")
def make_report(
    ctx: typer.Context,
    scenario: Optional[int] = typer.Option(
        None,
        "--scenario",
        "-s",
        help="Scenario number to generate report for",
    ),
    params: Optional[str] = typer.Option(
        None,
        "--params",
        "-p",
        help="Specific parameters as JSON array",
    ),
):
    """
    Generate HTML reports from existing simulation data.
    
    Examples:
        slacgs make-report --scenario 1
        slacgs make-report --params "[1,4,0.6]"
    """
    logger = get_logger("slacgs.cli.make_report")
    
    if not scenario and not params:
        console.print("[red]✗[/red] Must specify either --scenario or --params", style="bold red")
        raise typer.Exit(1)
    
    console.print(f"\n[bold cyan]Generating report[/bold cyan]\n")
    
    try:
        if scenario:
            console.print(f"Scenario: [yellow]{scenario}[/yellow]")
            
            from slacgs.demo import SCENARIOS
            from slacgs.reporting.report_utils import create_scenario_report
            
            if scenario < 1 or scenario > len(SCENARIOS):
                console.print(f"[red]✗[/red] Invalid scenario number. Must be 1-{len(SCENARIOS)}", style="bold red")
                raise typer.Exit(1)
            
            scenario_params = SCENARIOS[scenario - 1]
            console.print(f"Parameters: {len(scenario_params)} simulations\n")
            
            with console.status(f"[bold green]Generating report...[/bold green]"):
                create_scenario_report(scenario_params, scenario)
            
            console.print(f"[bold green]✓[/bold green] Report generated: scenario_{scenario}_report.html\n")
            
        else:  # params
            import json
            param_list = json.loads(params)
            console.print(f"Parameters: [yellow]{param_list}[/yellow]\n")
            
            # Import and create report for specific simulation
            console.print("[yellow]Note:[/yellow] Individual simulation reports are auto-generated during run-simulation\n")
            
    except Exception as e:
        console.print(f"[red]✗[/red] Report generation failed: {e}", style="bold red")
        logger.error(f"Report error: {e}", exc_info=True)
        raise typer.Exit(1)


@app.command()
def publish(
    ctx: typer.Context,
    auto_push: bool = typer.Option(
        False,
        "--auto-push",
        help="Automatically push to GitHub after publishing",
    ),
    branch: str = typer.Option(
        "reports-pages",
        "--branch",
        "-b",
        help="Target branch for GitHub Pages",
    ),
):
    """
    Publish reports to GitHub Pages.
    
    Examples:
        slacgs publish
        slacgs publish --auto-push
    """
    logger = get_logger("slacgs.cli.publish")
    
    console.print(f"\n[bold cyan]Publishing to GitHub Pages[/bold cyan]\n")
    console.print(f"Target branch: [yellow]{branch}[/yellow]\n")
    
    try:
        from slacgs.publish.publisher import publish_to_pages
        
        with console.status(f"[bold green]Publishing...[/bold green]"):
            publish_to_pages(auto_push=auto_push, branch=branch)
        
        console.print(f"[bold green]✓[/bold green] Published successfully!")
        
        if auto_push:
            console.print("Changes pushed to GitHub\n")
        else:
            console.print(f"[yellow]Note:[/yellow] Run 'git push origin {branch}' to deploy\n")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Publishing failed: {e}", style="bold red")
        logger.error(f"Publishing error: {e}", exc_info=True)
        raise typer.Exit(1)


# Config subcommands
config_app = typer.Typer(help="Configuration management commands")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(ctx: typer.Context):
    """Show current configuration."""
    config = ctx.obj["config"]
    
    console.print("\n[bold cyan]Current Configuration[/bold cyan]\n")
    
    # Create table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Section", style="cyan")
    table.add_column("Key", style="yellow")
    table.add_column("Value", style="green")
    
    for section, values in config.items():
        if section.startswith("_"):
            continue
        if isinstance(values, dict):
            for key, value in values.items():
                table.add_row(section, key, str(value))
        else:
            table.add_row("", section, str(values))
    
    console.print(table)
    console.print()


@config_app.command("init")
def config_init(
    project: bool = typer.Option(
        True,
        "--project/--user",
        help="Create project config (./slacgs.toml) or user config (~/.config/slacgs/config.toml)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing config file",
    ),
):
    """Create a configuration file template."""
    console.print(f"\n[bold cyan]Initializing configuration[/bold cyan]\n")
    
    try:
        if project:
            path = Path("./slacgs.toml")
            init_project_config(path, force=force)
            console.print(f"[bold green]SUCCESS:[/bold green] Created: [cyan]{path.absolute()}[/cyan]")
        else:
            from slacgs.config import init_user_config
            path = init_user_config(force=force)
            console.print(f"[bold green]SUCCESS:[/bold green] Created: [cyan]{path}[/cyan]")
        
        console.print("\nEdit the file to customize your configuration.\n")
        
    except FileExistsError:
        console.print(f"[red]✗[/red] Config file already exists. Use --force to overwrite.", style="bold red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to create config: {e}", style="bold red")
        raise typer.Exit(1)


@config_app.command("validate")
def config_validate(
    config_file: Optional[str] = typer.Option(
        None,
        "--file",
        "-f",
        help="Config file to validate",
    ),
):
    """Validate configuration file."""
    console.print(f"\n[bold cyan]Validating configuration[/bold cyan]\n")
    
    try:
        if config_file:
            config = load_config(config_path=Path(config_file))
        else:
            config = load_config()
        
        validate_config(config)
        
        console.print(f"[bold green]SUCCESS:[/bold green] Configuration is valid\n")
        
    except ConfigError as e:
        console.print(f"[red]✗[/red] Validation failed: {e}", style="bold red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]✗[/red] Error: {e}", style="bold red")
        raise typer.Exit(1)


@app.command(name="cleanup-logs")
def cleanup_logs(
    ctx: typer.Context,
    older_than_days: int = typer.Option(
        30,
        "--older-than",
        "-d",
        help="Delete log files older than N days",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be deleted without actually deleting",
    ),
):
    """
    Clean up old log files.
    
    Deletes log files older than the specified number of days.
    Uses the configured output directory to find logs.
    
    Examples:
        slacgs cleanup-logs --older-than 30  # Delete logs older than 30 days
        slacgs cleanup-logs --dry-run        # See what would be deleted
    """
    logger = get_logger("slacgs.cli.cleanup_logs")
    config = ctx.obj["config"]
    
    from datetime import datetime, timedelta
    from slacgs.config import get_output_dir
    
    console.print(f"\n[bold cyan]Cleaning up log files[/bold cyan]\n")
    
    try:
        output_dir = get_output_dir(config)
        logs_dir = output_dir / "logs"
        
        if not logs_dir.exists():
            console.print(f"[yellow]No logs directory found at:[/yellow] {logs_dir}\n")
            return
        
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        deleted_count = 0
        deleted_size = 0
        
        console.print(f"Scanning: [cyan]{logs_dir}[/cyan]")
        console.print(f"Cutoff date: [yellow]{cutoff_date.strftime('%Y-%m-%d %H:%M:%S')}[/yellow]\n")
        
        # Find all .log files
        log_files = list(logs_dir.glob("*.log*"))
        
        if not log_files:
            console.print("[yellow]No log files found[/yellow]\n")
            return
        
        for log_file in log_files:
            # Get file modification time
            mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            
            if mtime < cutoff_date:
                size = log_file.stat().st_size
                size_mb = size / (1024 * 1024)
                
                if dry_run:
                    console.print(f"  [dim]Would delete:[/dim] {log_file.name} ({size_mb:.2f} MB) - {mtime.strftime('%Y-%m-%d')}")
                else:
                    console.print(f"  [red]Deleting:[/red] {log_file.name} ({size_mb:.2f} MB) - {mtime.strftime('%Y-%m-%d')}")
                    log_file.unlink()
                    logger.info(f"Deleted log file: {log_file}")
                
                deleted_count += 1
                deleted_size += size
        
        if deleted_count > 0:
            size_mb = deleted_size / (1024 * 1024)
            if dry_run:
                console.print(f"\n[yellow]Would delete {deleted_count} file(s), {size_mb:.2f} MB total[/yellow]\n")
            else:
                console.print(f"\n[green]Deleted {deleted_count} file(s), {size_mb:.2f} MB total[/green]\n")
        else:
            console.print(f"\n[green]No log files older than {older_than_days} days[/green]\n")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Cleanup failed: {e}", style="bold red")
        logger.error(f"Log cleanup error: {e}", exc_info=True)
        raise typer.Exit(1)


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()

