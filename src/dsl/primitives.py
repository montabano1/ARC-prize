from typing import List, Dict, Any, Callable
import numpy as np
from dataclasses import dataclass

@dataclass
class DSLPrimitive:
    """A primitive operation in the DSL"""
    name: str
    operation: Callable
    parameters: List[str]
    description: str
    complexity: float
    usage_count: int = 0
    success_rate: float = 0.0

class DynamicPrimitiveLibrary:
    """Dynamic library of DSL primitives that can evolve over time"""
    def __init__(self):
        self.primitives: Dict[str, DSLPrimitive] = {}
        self._initialize_base_primitives()
        
    def _initialize_base_primitives(self):
        """Initialize the basic set of primitives"""
        # Border operations
        def get_border(grid):
            """Get a mask of border cells"""
            border = np.zeros_like(grid, dtype=bool)
            border[0, :] = border[-1, :] = True  # Top and bottom rows
            border[:, 0] = border[:, -1] = True  # Left and right columns
            return border
            
        def replace_border(grid, old_val, new_val):
            """Replace border values"""
            border = get_border(grid)
            result = grid.copy()
            result[border & (grid == old_val)] = new_val
            return result
            
        self.add_primitive(
            "get_border",
            get_border,
            [],
            "Get a mask of border cells",
            0.2
        )
        
        self.add_primitive(
            "replace_border",
            replace_border,
            ["old_val", "new_val"],
            "Replace border values that match old_val with new_val",
            0.3
        )
        
        # Color operations
        self.add_primitive(
            "invert",
            lambda grid: 1 - grid,
            [],
            "Invert all values in the grid",
            0.1
        )
        
        # Geometric operations
        self.add_primitive(
            "rotate90",
            lambda grid: np.rot90(grid),
            [],
            "Rotate grid 90 degrees clockwise",
            0.2
        )
        
        self.add_primitive(
            "mirror",
            lambda grid: np.fliplr(grid),
            [],
            "Mirror the grid horizontally",
            0.2
        )
        
        self.add_primitive(
            "fill",
            lambda grid, value: np.full_like(grid, value),
            ["value"],
            "Fill entire grid with a value",
            0.1
        )
        
        # Pattern operations
        self.add_primitive(
            "get_border",
            lambda grid: np.pad(grid, ((1,1), (1,1)), 'constant', constant_values=0)[1:-1,1:-1],
            [],
            "Get the border cells of the grid",
            0.3
        )
        
        self.add_primitive(
            "get_interior",
            lambda grid: grid[1:-1,1:-1],
            [],
            "Get the interior cells of the grid",
            0.3
        )
        
        # Color region operations
        self.add_primitive(
            "flood_fill",
            lambda grid, x, y, value: self._flood_fill(grid.copy(), x, y, grid[x,y], value),
            ["x", "y", "value"],
            "Fill connected region starting at (x,y) with value",
            0.4
        )
        
        self.add_primitive(
            "replace_color",
            lambda grid, old_val, new_val: np.where(grid == old_val, new_val, grid),
            ["old_val", "new_val"],
            "Replace all occurrences of old_val with new_val",
            0.2
        )
        
        # Shape operations
        self.add_primitive(
            "extract_shape",
            lambda grid, value: np.where(grid == value, 1, 0),
            ["value"],
            "Extract binary mask of cells with given value",
            0.3
        )
        
        self.add_primitive(
            "apply_mask",
            lambda grid, mask, value: np.where(mask == 1, value, grid),
            ["mask", "value"],
            "Apply value to grid where mask is 1",
            0.3
        )

    def _flood_fill(self, grid, x, y, old_val, new_val):
        """Helper function for flood fill operation"""
        if x < 0 or x >= grid.shape[0] or y < 0 or y >= grid.shape[1]:
            return grid
        if grid[x,y] != old_val:
            return grid
            
        grid[x,y] = new_val
        self._flood_fill(grid, x+1, y, old_val, new_val)
        self._flood_fill(grid, x-1, y, old_val, new_val)
        self._flood_fill(grid, x, y+1, old_val, new_val)
        self._flood_fill(grid, x, y-1, old_val, new_val)
        return grid

    def add_primitive(self, name: str, operation: Callable, 
                     parameters: List[str], description: str, 
                     complexity: float) -> None:
        """Add a new primitive to the library"""
        self.primitives[name] = DSLPrimitive(
            name=name,
            operation=operation,
            parameters=parameters,
            description=description,
            complexity=complexity
        )
    
    def get_primitive(self, name: str) -> DSLPrimitive:
        """Get a primitive by name"""
        return self.primitives.get(name)
    
    def update_primitive_stats(self, name: str, success: bool) -> None:
        """Update usage statistics for a primitive"""
        if name in self.primitives:
            primitive = self.primitives[name]
            primitive.usage_count += 1
            if success:
                # Update success rate using exponential moving average
                alpha = 0.1  # Learning rate
                primitive.success_rate = (1 - alpha) * primitive.success_rate + alpha
    
    def generate_new_primitive(self, base_primitives: List[str], 
                             composition_rule: str) -> DSLPrimitive:
        """Generate a new primitive by composing existing ones"""
        # This will be implemented with LLM assistance
        pass

    def prune_underperforming(self, min_usage: int = 10, 
                             min_success_rate: float = 0.3) -> None:
        """Remove primitives that are not performing well"""
        to_remove = []
        for name, primitive in self.primitives.items():
            if (primitive.usage_count >= min_usage and 
                primitive.success_rate < min_success_rate):
                to_remove.append(name)
        
        for name in to_remove:
            del self.primitives[name]

    def list_primitives(self) -> List[str]:
        """List all available primitives"""
        return list(self.primitives.keys())
        
    def execute_primitive(self, name: str, grid: np.ndarray, 
                         params: Dict[str, Any] = None) -> np.ndarray:
        """Execute a primitive operation on a grid"""
        if name not in self.primitives:
            raise ValueError(f"Unknown primitive: {name}")
            
        primitive = self.primitives[name]
        params = params or {}
        
        # Validate parameters
        for param in primitive.parameters:
            if param not in params:
                raise ValueError(f"Missing parameter {param} for primitive {name}")
                
        # Execute operation
        if primitive.parameters:
            result = primitive.operation(grid, **params)
        else:
            result = primitive.operation(grid)
            
        # Update usage statistics
        primitive.usage_count += 1
        
        return result
