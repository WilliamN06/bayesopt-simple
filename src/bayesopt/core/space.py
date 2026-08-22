"""Parameter space definition and transformation."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum


class ParameterType(Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    LOG = "log"
    CATEGORICAL = "categorical"


class ParameterSpace:
    """Defines and manages the parameter search space."""
    
    def __init__(self, bounds: Dict[str, Tuple]):
        if not bounds:
            raise ValueError("Bounds dictionary cannot be empty")
            
        self.bounds = bounds
        self.param_names = list(bounds.keys())
        self.n_params = len(bounds)
        
        self.param_types = {}
        self.param_ranges = {}
        self.categorical_mappings = {}
        self.dim_indices = {}
        self.original_ranges = {}
        
        for idx, (name, spec) in enumerate(bounds.items()):
            self.dim_indices[name] = idx
            parsed = self._parse_spec(name, spec)
            self.param_types[name] = parsed['type']
            
            if parsed['type'] == ParameterType.CATEGORICAL:
                self.categorical_mappings[name] = parsed['categories']
                self.original_ranges[name] = (0, len(parsed['categories']) - 1)
            else:
                self.param_ranges[name] = parsed['range']
                self.original_ranges[name] = parsed['range']
        
        self._update_bounds_arrays()
    
    def _parse_spec(self, name: str, spec: Tuple) -> Dict:
        if all(isinstance(item, str) for item in spec):
            categories = list(spec)
            if len(set(categories)) != len(categories):
                raise ValueError(f"Categorical parameter '{name}' has duplicate categories: {categories}")
            return {'type': ParameterType.CATEGORICAL, 'categories': categories}
        
        if len(spec) == 2:
            low, high = spec
            if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                raise ValueError(f"Parameter '{name}': low and high must be numbers")
            if low >= high:
                raise ValueError(f"Parameter '{name}': low ({low}) must be < high ({high})")
            return {'type': ParameterType.CONTINUOUS, 'range': (float(low), float(high))}
        
        if len(spec) == 3:
            if spec[2] == 'int':
                low, high, _ = spec
                if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                    raise ValueError(f"Integer parameter '{name}': low and high must be numbers")
                if low >= high:
                    raise ValueError(f"Integer parameter '{name}': low ({low}) must be < high ({high})")
                return {'type': ParameterType.INTEGER, 'range': (int(low), int(high))}
            elif spec[2] == 'log':
                low, high, _ = spec
                if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                    raise ValueError(f"Log-scale parameter '{name}': low and high must be numbers")
                if low <= 0:
                    raise ValueError(f"Log-scale parameter '{name}': low ({low}) must be > 0")
                if low >= high:
                    raise ValueError(f"Log-scale parameter '{name}': low ({low}) must be < high ({high})")
                return {'type': ParameterType.LOG, 'range': (float(low), float(high))}
            else:
                categories = list(spec)
                if len(set(categories)) != len(categories):
                    raise ValueError(f"Categorical parameter '{name}' has duplicate categories: {categories}")
                return {'type': ParameterType.CATEGORICAL, 'categories': categories}
        else:
            categories = list(spec)
            if len(set(categories)) != len(categories):
                raise ValueError(f"Categorical parameter '{name}' has duplicate categories: {categories}")
            return {'type': ParameterType.CATEGORICAL, 'categories': categories}
    
    def _update_bounds_arrays(self):
        self.lows = np.zeros(self.n_params)
        self.highs = np.zeros(self.n_params)
        self.types = []
        
        for idx, name in enumerate(self.param_names):
            ptype = self.param_types[name]
            self.types.append(ptype)
            
            if ptype == ParameterType.CATEGORICAL:
                self.lows[idx] = 0
                self.highs[idx] = 1
            elif ptype == ParameterType.INTEGER:
                low, high = self.param_ranges[name]
                self.lows[idx] = 0
                self.highs[idx] = 1
            else:
                low, high = self.param_ranges[name]
                self.lows[idx] = low
                self.highs[idx] = high
    
    def sample_random(self, n_points: int = 1) -> Union[Dict, List[Dict]]:
        if n_points == 1:
            return self._sample_single()
        else:
            return [self._sample_single() for _ in range(n_points)]
    
    def _sample_single(self) -> Dict:
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
                log_low, log_high = np.log(low), np.log(high)
                params[name] = np.exp(np.random.uniform(log_low, log_high))
            elif ptype == ParameterType.CATEGORICAL:
                categories = self.categorical_mappings[name]
                params[name] = np.random.choice(categories)
        
        return params
    
    def to_array(self, params: Dict[str, Any]) -> np.ndarray:
        arr = np.zeros(self.n_params)
        
        for name, value in params.items():
            if name not in self.dim_indices:
                raise KeyError(f"Unknown parameter: {name}")
            
            idx = self.dim_indices[name]
            ptype = self.param_types[name]
            
            if ptype == ParameterType.CONTINUOUS:
                low, high = self.param_ranges[name]
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
                    raise ValueError(f"Value {value} not in categories {categories}")
                cat_idx = categories.index(value)
                arr[idx] = cat_idx / (len(categories) - 1) if len(categories) > 1 else 0.0
        
        return arr
    
    def from_array(self, arr: np.ndarray) -> Dict[str, Any]:
        if len(arr) != self.n_params:
            raise ValueError(f"Array length {len(arr)} does not match number of parameters {self.n_params}")
        
        params = {}
        
        for name in self.param_names:
            idx = self.dim_indices[name]
            ptype = self.param_types[name]
            value = np.clip(arr[idx], 0.0, 1.0)
            
            if ptype == ParameterType.CONTINUOUS:
                low, high = self.param_ranges[name]
                params[name] = value * (high - low) + low
            elif ptype == ParameterType.INTEGER:
                low, high = self.param_ranges[name]
                raw = value * (high - low) + low
                params[name] = int(np.clip(np.round(raw), low, high))
            elif ptype == ParameterType.LOG:
                low, high = self.param_ranges[name]
                log_low, log_high = np.log(low), np.log(high)
                params[name] = np.exp(value * (log_high - log_low) + log_low)
            elif ptype == ParameterType.CATEGORICAL:
                categories = self.categorical_mappings[name]
                cat_idx = int(np.round(value * (len(categories) - 1)))
                cat_idx = np.clip(cat_idx, 0, len(categories) - 1)
                params[name] = categories[cat_idx]
        
        return params
    
    def bounds_array(self) -> np.ndarray:
        lows = np.zeros(self.n_params)
        highs = np.ones(self.n_params)
        
        for idx, name in enumerate(self.param_names):
            ptype = self.param_types[name]
            
            if ptype == ParameterType.CATEGORICAL:
                lows[idx] = 0
                highs[idx] = 1
            elif ptype == ParameterType.INTEGER:
                lows[idx] = 0
                highs[idx] = 1
            else:
                low, high = self.param_ranges[name]
                lows[idx] = low
                highs[idx] = high
        
        return np.column_stack([lows, highs])
    
    def get_dimensions(self) -> int:
        return self.n_params
    
    def get_names(self) -> List[str]:
        return self.param_names.copy()
    
    def get_default(self) -> Dict[str, Any]:
        default_arr = np.ones(self.n_params) * 0.5
        return self.from_array(default_arr)
    
    def is_categorical(self, name: str) -> bool:
        return self.param_types[name] == ParameterType.CATEGORICAL