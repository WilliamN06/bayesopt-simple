"""State management for Bayesian optimisation."""

from typing import Dict, List, Any, Optional
import time
import numpy as np


class OptimizerState:
    """Tracks the state of the optimisation process."""
    
    def __init__(self):
        self.history: List[Dict] = []
        self.best_x: Optional[Dict] = None
        self.best_y: float = float('inf')
        self.iteration: int = 0
        self.start_time: Optional[float] = None
        self.elapsed_time: float = 0.0
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.converged: bool = False
        self.exploration_exploitation_balance: float = 0.5
    
    def start(self) -> None:
        """Start timing the optimisation."""
        self.start_time = time.time()
    
    def update(self, x: Dict, y: float) -> None:
        """Update state with a new evaluation."""
        self.iteration += 1
        self.elapsed_time = time.time() - self.start_time if self.start_time else 0.0
        
        entry = {
            'iteration': self.iteration,
            'params': x.copy(),
            'value': y,
            'timestamp': self.elapsed_time
        }
        self.history.append(entry)
        
        if y < self.best_y:
            self.best_y = y
            self.best_x = x.copy()
    
    def get_best(self) -> tuple:
        """Get best parameters and value."""
        return self.best_x, self.best_y
    
    def get_history(self) -> List[Dict]:
        """Get full history."""
        return self.history.copy()
    
    def get_X_y(self) -> tuple:
        """Get training data as arrays."""
        if not self.history:
            return np.array([]), np.array([])
        
        X = []
        y = []
        for entry in self.history:
            X.append(list(entry['params'].values()))
            y.append(entry['value'])
        return np.array(X), np.array(y)
    
    def get_iteration(self) -> int:
        """Get current iteration number."""
        return self.iteration
    
    def is_empty(self) -> bool:
        """Check if any evaluations have been made."""
        return len(self.history) == 0
    
    def reset(self) -> None:
        """Reset state for new optimisation."""
        self.history = []
        self.best_x = None
        self.best_y = float('inf')
        self.iteration = 0
        self.start_time = None
        self.elapsed_time = 0.0
        self.X_train = None
        self.y_train = None
        self.converged = False
        self.exploration_exploitation_balance = 0.5