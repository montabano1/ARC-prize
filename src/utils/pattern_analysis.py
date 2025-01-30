import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Pattern:
    grid: np.ndarray
    position: Tuple[int, int]
    size: Tuple[int, int]
    frequency: int
    symmetry: float
    complexity: float

class PatternAnalyzer:
    def __init__(self, min_size: int = 2, max_size: int = 5):
        self.min_size = min_size
        self.max_size = max_size
        
    def find_patterns(self, grid: np.ndarray) -> List[Pattern]:
        """Find interesting patterns in the grid."""
        patterns = []
        
        # Look for patterns of different sizes
        for h in range(self.min_size, min(self.max_size, grid.shape[0] + 1)):
            for w in range(self.min_size, min(self.max_size, grid.shape[1] + 1)):
                patterns.extend(self._find_patterns_of_size(grid, h, w))
        
        # Sort patterns by complexity and frequency
        patterns.sort(key=lambda p: (p.complexity * p.frequency, p.symmetry), reverse=True)
        return patterns[:10]  # Return top 10 patterns
        
    def _find_patterns_of_size(self, grid: np.ndarray, height: int, width: int) -> List[Pattern]:
        """Find all patterns of a specific size in the grid."""
        patterns = {}
        
        for i in range(grid.shape[0] - height + 1):
            for j in range(grid.shape[1] - width + 1):
                sub_grid = grid[i:i+height, j:j+width]
                pattern_key = sub_grid.tobytes()
                
                if pattern_key not in patterns:
                    symmetry = self._calculate_symmetry(sub_grid)
                    complexity = self._calculate_complexity(sub_grid)
                    patterns[pattern_key] = Pattern(
                        grid=sub_grid,
                        position=(i, j),
                        size=(height, width),
                        frequency=1,
                        symmetry=symmetry,
                        complexity=complexity
                    )
                else:
                    patterns[pattern_key].frequency += 1
        
        return list(patterns.values())
        
    def _calculate_symmetry(self, grid: np.ndarray) -> float:
        """Calculate symmetry score for a grid."""
        h_sym = self._horizontal_symmetry(grid)
        v_sym = self._vertical_symmetry(grid)
        d_sym = self._diagonal_symmetry(grid)
        return max(h_sym, v_sym, d_sym)
        
    def _horizontal_symmetry(self, grid: np.ndarray) -> float:
        """Calculate horizontal symmetry score."""
        rows = grid.shape[0]
        matches = sum(
            np.array_equal(grid[i, :], grid[rows-1-i, :])
            for i in range(rows//2)
        )
        return matches / (rows//2) if rows > 1 else 1.0
        
    def _vertical_symmetry(self, grid: np.ndarray) -> float:
        """Calculate vertical symmetry score."""
        cols = grid.shape[1]
        matches = sum(
            np.array_equal(grid[:, j], grid[:, cols-1-j])
            for j in range(cols//2)
        )
        return matches / (cols//2) if cols > 1 else 1.0
        
    def _diagonal_symmetry(self, grid: np.ndarray) -> float:
        """Calculate diagonal symmetry score."""
        if grid.shape[0] != grid.shape[1]:
            return 0.0
            
        n = grid.shape[0]
        main_diag = sum(
            np.array_equal(grid[i, i], grid[n-1-i, n-1-i])
            for i in range(n//2)
        )
        anti_diag = sum(
            np.array_equal(grid[i, n-1-i], grid[n-1-i, i])
            for i in range(n//2)
        )
        return max(main_diag, anti_diag) / (n//2) if n > 1 else 1.0
        
    def _calculate_complexity(self, grid: np.ndarray) -> float:
        """Calculate pattern complexity score."""
        # Consider number of unique values
        unique_ratio = len(np.unique(grid)) / grid.size
        
        # Consider entropy
        values, counts = np.unique(grid, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities))
        max_entropy = np.log2(grid.size)  # Maximum possible entropy
        entropy_ratio = entropy / max_entropy if max_entropy > 0 else 0
        
        # Combine metrics
        return 0.4 * unique_ratio + 0.6 * entropy_ratio
        
    def find_transformations(self, input_grid: np.ndarray, 
                           output_grid: np.ndarray) -> List[Dict]:
        """Find possible transformations between input and output patterns."""
        transformations = []
        
        # Check for color inversion
        if np.array_equal(1 - input_grid, output_grid):
            transformations.append({
                'type': 'color_inversion',
                'confidence': 1.0
            })
        
        # Check for border addition
        if self._is_border_addition(input_grid, output_grid):
            transformations.append({
                'type': 'border_addition',
                'confidence': 1.0
            })
        
        # Check for pattern replication
        if self._is_pattern_replication(input_grid, output_grid):
            transformations.append({
                'type': 'pattern_replication',
                'confidence': 0.8
            })
        
        return transformations
        
    def _is_border_addition(self, input_grid: np.ndarray, 
                          output_grid: np.ndarray) -> bool:
        """Check if transformation is border addition."""
        if output_grid.shape[0] != input_grid.shape[0] + 2 or \
           output_grid.shape[1] != input_grid.shape[1] + 2:
            return False
            
        inner_region = output_grid[1:-1, 1:-1]
        return np.array_equal(inner_region, input_grid)
        
    def _is_pattern_replication(self, input_grid: np.ndarray,
                              output_grid: np.ndarray) -> bool:
        """Check if transformation involves pattern replication."""
        h_ratio = output_grid.shape[0] / input_grid.shape[0]
        w_ratio = output_grid.shape[1] / input_grid.shape[1]
        
        if not (h_ratio.is_integer() and w_ratio.is_integer()):
            return False
            
        h_ratio = int(h_ratio)
        w_ratio = int(w_ratio)
        
        # Check if output is made up of repeated input
        for i in range(h_ratio):
            for j in range(w_ratio):
                region = output_grid[
                    i*input_grid.shape[0]:(i+1)*input_grid.shape[0],
                    j*input_grid.shape[1]:(j+1)*input_grid.shape[1]
                ]
                if not np.array_equal(region, input_grid):
                    return False
                    
        return True
