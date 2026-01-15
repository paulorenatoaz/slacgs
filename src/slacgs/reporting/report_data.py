"""
ReportData - Lightweight data container for simulation results.

This module provides a pure data object that replaces passing the entire
Simulator instance to Report, breaking the circular dependency and enabling
better serialization, testing, and memory efficiency.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
from datetime import datetime


@dataclass
class ReportData:
    """
    Data container for simulation results.
    
    This is a pure data transfer object (DTO) with no business logic.
    It contains only the data that Report needs from Simulator and Model.
    
    Attributes:
        dims: List of dimensionalities to simulate, e.g., [1, 2]
        loss_types: Loss types to compute, e.g., ['THEORETICAL', 'EMPIRICAL_TRAIN', 'EMPIRICAL_TEST']
        test_mode: True if running in test mode (faster, less precise)
        test_samples_amt: Number of test samples (default: 1024)
        full_n_range: Whether to use full N range even in test mode
        iters_per_step: Iterations per step (5 in test mode, 10 otherwise)
        max_steps: Maximum steps (20 in test mode, 100 otherwise)
        min_steps: Minimum steps (20 in test mode, 100 otherwise)
        params: Model parameters, e.g., [1, 4, 0.6] for 2D
        N: Sample sizes, e.g., [2, 4, 8, 16, 32, ...]
        sigma: Standard deviations
        rho: Correlations
        dim: Model dimensionality (2 or 3)
        mean_pos: Mean of positive class
        mean_neg: Mean of negative class
        dictionary: Dictionary type (e.g., ['LINEAR'])
        rho_matrix: Correlation matrix
        max_n: Maximum sample size
        cov: Covariance matrix
        iter_N: Iterations per N for each dim and loss type
        loss_N: Loss values per N for each dim and loss type
        loss_bayes: Bayes error rate per dim
        d: Distance metric per dim
        duration: Total simulation duration in hours
        time_spent: Time tracking dict
        simulation_id: Optional ID from JSON
        timestamp: ISO format timestamp
    """
    # Simulation configuration (required)
    dims: List[int]
    loss_types: List[str]
    test_mode: bool
    
    # Model configuration (required)
    params: List[float]
    N: List[int]
    sigma: List[float]
    rho: List[float]
    dim: int
    
    # Simulation configuration (optional with defaults)
    test_samples_amt: int = 1024
    full_n_range: bool = False
    iters_per_step: int = 5
    max_steps: int = 20
    min_steps: int = 20
    
    # Model configuration (optional with defaults)
    mean_pos: List[float] = field(default_factory=list)
    mean_neg: List[float] = field(default_factory=list)
    dictionary: List[str] = field(default_factory=lambda: ['LINEAR'])
    rho_matrix: List[List[Optional[float]]] = field(default_factory=list)
    max_n: int = 8192
    cov: List[List[float]] = field(default_factory=list)
    
    # Simulation results (populated during run)
    iter_N: Dict[int, Dict[str, List[float]]] = field(default_factory=dict)
    loss_N: Dict[int, Dict[str, List[float]]] = field(default_factory=dict)
    loss_bayes: Dict[int, float] = field(default_factory=dict)
    d: Dict[int, float] = field(default_factory=dict)
    duration: float = 0.0
    time_spent: Dict[str, Any] = field(default_factory=dict)
    
    # Optional metadata
    simulation_id: Optional[int] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        
        Returns:
            Dictionary representation of ReportData
        """
        return asdict(self)
    
    def to_json(self, filepath: Path) -> None:
        """
        Save to JSON file.
        
        Args:
            filepath: Path to save JSON file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReportData':
        """
        Create ReportData from dictionary (e.g., loaded from JSON).
        
        Args:
            data: Dictionary with ReportData fields
            
        Returns:
            ReportData instance
        """
        return cls(**data)
    
    @classmethod
    def from_json(cls, filepath: Path) -> 'ReportData':
        """
        Load ReportData from JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            ReportData instance
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def validate(self) -> bool:
        """
        Validate data integrity.
        
        Returns:
            True if validation passes
            
        Raises:
            AssertionError: If validation fails
        """
        # Check required fields are non-empty
        assert len(self.dims) > 0, "dims cannot be empty"
        assert len(self.loss_types) > 0, "loss_types cannot be empty"
        assert len(self.N) > 0, "N cannot be empty"
        assert len(self.params) > 0, "params cannot be empty"
        assert self.dim in [2, 3], f"dim must be 2 or 3, got {self.dim}"
        
        # Check parameter count matches dimensionality
        expected_params = 3 if self.dim == 2 else 6
        assert len(self.params) == expected_params, \
            f"{self.dim}D model needs {expected_params} params, got {len(self.params)}"
        
        # Check sigma and rho lengths
        assert len(self.sigma) == self.dim, \
            f"sigma length ({len(self.sigma)}) must match dim ({self.dim})"
        
        expected_rho = (self.dim * (self.dim - 1)) // 2
        assert len(self.rho) == expected_rho, \
            f"{self.dim}D model needs {expected_rho} rho values, got {len(self.rho)}"
        
        # Check dims list contains valid values
        for d in self.dims:
            assert 1 <= d <= self.dim, \
                f"dim value {d} must be between 1 and {self.dim}"
        
        return True
    
    def summary(self) -> str:
        """
        Generate human-readable summary of the data.
        
        Returns:
            Multi-line summary string
        """
        return f"""ReportData Summary:
  Model: {self.dim}D, params={self.params}
  Dimensions simulated: {self.dims}
  Sample sizes (N): {len(self.N)} values from {min(self.N)} to {max(self.N)}
  Loss types: {', '.join(self.loss_types)}
  Test mode: {self.test_mode}
  Duration: {self.duration:.4f} hours
  Timestamp: {self.timestamp}
"""
