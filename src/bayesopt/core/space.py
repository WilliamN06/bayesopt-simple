"""Parameter space definition and transformation."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum


class ParameterType(Enum):
    """Types of parameters supported."""
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    LOG = "log"  # Log-scale continuous
    CATEGORICAL = "categorical"


class ParameterSpace:
    """
    Defines and manages the parameter search space.

    """
    
    def __init__(self, bounds: Dict[str, Tuple]):
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.n_params = len(bounds)
        
        # Parse and store parameter information
        self.param_types = {}
        self.param_ranges = {}
        self.categorical_mappings = {}
        self.dim_indices = {}
        
        for idx, (name, spec) in enumerate(bounds.items()):
            self.dim_indices[name] = idx
            parsed = self._parse_spec(name, spec)
            self.param_types[name] = parsed['type']
            self.param_ranges[name] = parsed['range']
            
            if parsed['type'] == ParameterType.CATEGORICAL:
                self.categorical_mappings[name] = parsed['categories']
        
        # Store as arrays for efficient computation
        self._update_bounds_arrays()
    
    def _parse_spec(self, name: str, spec: Tuple) -> Dict:
        """
        Parse parameter specification.
        """
        if len(spec) == 2:
            # Continuous: (low, high)
            low, high = spec
            if low >= high:
                raise ValueError(f"Parameter '{name}': low ({low}) must be < high ({high})")
            return {'type': ParameterType.CONTINUOUS, 'range': (float(low), float(high))}
        
        elif len(spec) == 3:
            # Check for type specifier: int or log
            if spec[2] == 'int':
                low, high, _ = spec
                if low >= high:
                    raise ValueError(f"Integer parameter '{name}': low ({low}) must be < high ({high})")
                return {'type': ParameterType.INTEGER, 'range': (int(low), int(high))}
            elif spec[2] == 'log':
                low, high, _ = spec
                if low <= 0:
                    raise ValueError(f"Log-scale parameter '{name}': low ({low}) must be > 0")
                if low >= high:
                    raise ValueError(f"Log-scale parameter '{name}': low ({low}) must be < high ({high})")
                return {'type': ParameterType.LOG, 'range': (float(low), float(high))}
            else:
                # Could be categorical with 3 categories
                return {'type': ParameterType.CATEGORICAL, 'categories': list(spec)}
        
        else:
            # Categorical with any number of categories
            # All items must be hashable and unique
            categories = list(spec)
            if len(set(categories)) != len(categories):
                raise ValueError(f"Categorical parameter '{name}' has duplicate categories: {categories}")
            return {'type': ParameterType.CATEGORICAL, 'categories': categories}
    
    def _update_bounds_arrays(self):
        """Update internal arrays for efficient computation."""
        self.lows = np.zeros(self.n_params)
        self.highs = np.zeros(self.n_params)
        self.types = []
        
        for idx, name in enumerate(self.param_names):
            ptype = self.param_types[name]
            self.types.append(ptype)
            
            if ptype == ParameterType.CATEGORICAL:
                # For categorical, use 0 to n_categories-1 in scaled space
                self.lows[idx] = 0
                self.highs[idx] = len(self.categorical_mappings[name]) - 1
            else:
                low, high = self.param_ranges[name]
                self.lows[idx] = low
                self.highs[idx] = high
    
    def sample_random(self, n_points: int = 1) -> Union[Dict, List[Dict]]:
        """
        Sample random points from the parameter space.
     
        """
        if n_points == 1:
            return self._sample_single()
        else:
            return [self._sample_single() for _ in range(n_points)]
    
    def _sample_single(self) -> Dict:
        """Sample a single random point."""
        params = {}
        
        for name in self.param_names:
            ptype = self.param_types[name]
            
            if ptype == ParameterType.CONTINUOUS:
                low, high = self.param_ranges[name]
                params[name] = np.random.uniform(low, high)
            
            elif ptype == ParameterType.INTEGER:
                low, high = self.param_ranges[name]
                params[name] = np.random.randint(low, high + 1)
            
            elif ptype == ParameterType.LOG:
                low, high = self.param_ranges[name]
                # Sample uniformly in log space
                log_low, log_high = np.log(low), np.log(high)
                params[name] = np.exp(np.random.uniform(log_low, log_high))
            
            elif ptype == ParameterType.CATEGORICAL:
                categories = self.categorical_mappings[name]
                params[name] = np.random.choice(categories)
        
        return params
    
    def to_array(self, params: Dict[str, Any]) -> np.ndarray:
        """
        Convert parameter dictionary to scaled numpy array.
        
        """
        arr = np.zeros(self.n_params)
        
        for name, value in params.items():
            if name not in self.dim_indices:
                raise KeyError(f"Unknown parameter: {name}")
            
            idx = self.dim_indices[name]
            ptype = self.param_types[name]
            
            if ptype == ParameterType.CONTINUOUS:
                low, high = self.param_ranges[name]
                # Clamp to bounds (with small tolerance)
                value = np.clip(value, low, high)
                arr[idx] = (value - low) / (high - low)
            
            elif ptype == ParameterType.INTEGER:
                low, high = self.param_ranges[name]
                value = int(np.clip(value, low, high))
                arr[idx] = (value - low) / (high - low)
            
            elif ptype == ParameterType.LOG:
                low, high = self.param_ranges[name]
                value = np.clip(value, low, high)
                log_low, log_high = np.log(low), np.log(high)
                arr[idx] = (np.log(value) - log_low) / (log_high - log_low)
            
            elif ptype == ParameterType.CATEGORICAL:
                categories = self.categorical_mappings[name]
                if value not in categories:
                    raise ValueError(f"Value {value} not in categories {categories} for parameter {name}")
                # Encode as integer index (0 to n-1) scaled to [0, 1]
                cat_idx = categories.index(value)
                arr[idx] = cat_idx / (len(categories) - 1) if len(categories) > 1 else 0.0
        
        return arr
    
    def from_array(self, arr: np.ndarray) -> Dict[str, Any]:
        """
        Convert scaled numpy array back to parameter dictionary.
        """
        if len(arr) != self.n_params:
            raise ValueError(f"Array length {len(arr)} does not match number of parameters {self.n_params}")
        
        params = {}
        
        for name in self.param_names:
            idx = self.dim_indices[name]
            ptype = self.param_types[name]
            value = arr[idx]
            
            # Clamp to [0, 1] to avoid numerical issues
            value = np.clip(value, 0.0, 1.0)
            
            if ptype == ParameterType.CONTINUOUS:
                low, high = self.param_ranges[name]
                params[name] = value * (high - low) + low
            
            elif ptype == ParameterType.INTEGER:
                low, high = self.param_ranges[name]
                # Round to nearest integer
                raw = value * (high - low) + low
                params[name] = int(np.round(raw))
                # Ensure within bounds
                params[name] = np.clip(params[name], low, high)
            
            elif ptype == ParameterType.LOG:
                low, high = self.param_ranges[name]
                log_low, log_high = np.log(low), np.log(high)
                params[name] = np.exp(value * (log_high - log_low) + log_low)
            
            elif ptype == ParameterType.CATEGORICAL:
                categories = self.categorical_mappings[name]
                # Scale back to category index
                cat_idx = int(np.round(value * (len(categories) - 1)))
                cat_idx = np.clip(cat_idx, 0, len(categories) - 1)
                params[name] = categories[cat_idx]
        
        return params
    
    def bounds_array(self) -> np.ndarray:
        """
        Return bounds as array for scipy optimiser.
        
     
        """
        return np.column_stack([self.lows, self.highs])
    
    def get_dimensions(self) -> int:
        """Return the number of parameters."""
        return self.n_params
    
    def get_names(self) -> List[str]:
        """Return the parameter names."""
        return self.param_names.copy()
    
    def get_default(self) -> Dict[str, Any]:
        """
        Get default parameters (midpoint of each parameter).
        
        Returns
        -------
        dict
            Default parameters.
        """
        # Use midpoint of bounds
        default_arr = np.ones(self.n_params) * 0.5
        return self.from_array(default_arr)
    
    def is_categorical(self, name: str) -> bool:
        """Check if a parameter is categorical."""
        return self.param_types[name] == ParameterType.CATEGORICAL