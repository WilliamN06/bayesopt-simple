"""Sampling utilities for Bayesian optimisation."""

import numpy as np
from typing import Optional


def latin_hypercube_sampling(
    bounds: np.ndarray,
    n_points: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Generate Latin Hypercube samples within given bounds."""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_dims = bounds.shape[0]
    samples = np.zeros((n_points, n_dims))
    
    for dim in range(n_dims):
        low, high = bounds[dim]
        intervals = np.linspace(low, high, n_points + 1)
        for i in range(n_points):
            samples[i, dim] = np.random.uniform(intervals[i], intervals[i + 1])
        np.random.shuffle(samples[:, dim])
    
    return samples


def random_sampling(
    bounds: np.ndarray,
    n_points: int,
    random_state: Optional[int] = None
) -> np.ndarray:
    """Generate random samples within given bounds."""
    if random_state is not None:
        np.random.seed(random_state)
    
    n_dims = bounds.shape[0]
    samples = np.random.uniform(
        bounds[:, 0],
        bounds[:, 1],
        size=(n_points, n_dims)
    )
    
    return samples