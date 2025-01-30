import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

def plot_grid(grid: np.ndarray, title: Optional[str] = None) -> None:
    """Plot a single grid."""
    plt.figure(figsize=(5, 5))
    plt.imshow(grid, cmap='tab20')
    plt.grid(True)
    plt.xticks(range(grid.shape[1]))
    plt.yticks(range(grid.shape[0]))
    if title:
        plt.title(title)
    plt.colorbar()

def plot_task(input_grid: np.ndarray, output_grid: np.ndarray,
              predicted_grid: Optional[np.ndarray] = None) -> None:
    """Plot input, output, and optionally predicted grid side by side."""
    n_plots = 3 if predicted_grid is not None else 2
    plt.figure(figsize=(5 * n_plots, 5))
    
    # Plot input
    plt.subplot(1, n_plots, 1)
    plt.imshow(input_grid, cmap='tab20')
    plt.grid(True)
    plt.title('Input')
    plt.xticks(range(input_grid.shape[1]))
    plt.yticks(range(input_grid.shape[0]))
    plt.colorbar()
    
    # Plot output
    plt.subplot(1, n_plots, 2)
    plt.imshow(output_grid, cmap='tab20')
    plt.grid(True)
    plt.title('Expected Output')
    plt.xticks(range(output_grid.shape[1]))
    plt.yticks(range(output_grid.shape[0]))
    plt.colorbar()
    
    # Plot prediction if available
    if predicted_grid is not None:
        plt.subplot(1, n_plots, 3)
        plt.imshow(predicted_grid, cmap='tab20')
        plt.grid(True)
        plt.title('Predicted Output')
        plt.xticks(range(predicted_grid.shape[1]))
        plt.yticks(range(predicted_grid.shape[0]))
        plt.colorbar()
    
    plt.tight_layout()

def plot_pattern_matches(grid: np.ndarray, 
                        matches: List[Tuple[slice, slice]], 
                        pattern: np.ndarray) -> None:
    """Plot a grid with highlighted pattern matches."""
    plt.figure(figsize=(10, 5))
    
    # Plot original grid
    plt.subplot(1, 2, 1)
    plt.imshow(grid, cmap='tab20')
    plt.grid(True)
    plt.title('Original Grid')
    
    # Plot grid with highlighted matches
    plt.subplot(1, 2, 2)
    highlighted = grid.astype(float)  # Convert to float for highlighting
    highlight_mask = np.zeros_like(highlighted)
    for match in matches:
        highlight_mask[match[0], match[1]] = 0.3
    plt.imshow(highlighted + highlight_mask, cmap='tab20')
    plt.grid(True)
    plt.title('Pattern Matches')
    
    # Add pattern as inset
    ax_inset = plt.axes([0.85, 0.6, 0.1, 0.1])
    ax_inset.imshow(pattern, cmap='tab20')
    ax_inset.set_title('Pattern')
    ax_inset.grid(True)
    
    plt.tight_layout()
