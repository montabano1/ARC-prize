from typing import Dict, Any, List
import numpy as np

def invert_values(input_grid: np.ndarray) -> np.ndarray:
    """Strategy that inverts all values in the grid (0->1, 1->0)."""
    return 1 - input_grid

def preserve_center(input_grid: np.ndarray) -> np.ndarray:
    """Strategy that preserves the center value and inverts surrounding values."""
    output = np.ones_like(input_grid)
    center = tuple(d // 2 for d in input_grid.shape)
    output[center] = input_grid[center]
    return output

def copy_pattern(input_grid: np.ndarray) -> np.ndarray:
    """Strategy that copies the input grid exactly."""
    return input_grid.copy()

def fill_pattern(input_grid: np.ndarray, value: int = 1) -> np.ndarray:
    """Strategy that fills the entire grid with a specified value."""
    return np.full_like(input_grid, value)

STRATEGY_REGISTRY = {
    'invert_values': {
        'function': invert_values,
        'requirements': ['input_grid'],
        'description': 'Inverts all values in the grid'
    },
    'preserve_center': {
        'function': preserve_center,
        'requirements': ['input_grid'],
        'description': 'Preserves center value and inverts surroundings'
    },
    'copy_pattern': {
        'function': copy_pattern,
        'requirements': ['input_grid'],
        'description': 'Copies the input pattern exactly'
    },
    'fill_pattern': {
        'function': fill_pattern,
        'requirements': ['input_grid'],
        'description': 'Fills the grid with a specified value'
    }
}
