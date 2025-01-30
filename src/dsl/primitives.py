from typing import List, Dict, Any, Callable, Optional
import numpy as np
from dataclasses import dataclass
import json
import time

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
    def __init__(self, llm=None):
        self.primitives: Dict[str, DSLPrimitive] = {}
        self.llm = llm
        self.primitive_history: List[Dict[str, Any]] = []
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
            
        # Create primitives
        primitives = [
            DSLPrimitive(
                name="get_border",
                operation=get_border,
                parameters=[],
                description="Get a mask of border cells",
                complexity=0.2
            ),
            DSLPrimitive(
                name="replace_border",
                operation=replace_border,
                parameters=["old_val", "new_val"],
                description="Replace border values that match old_val with new_val",
                complexity=0.3
            ),
            DSLPrimitive(
                name="invert",
                operation=lambda grid: 1 - grid,
                parameters=[],
                description="Invert all values in the grid",
                complexity=0.1
            ),
            DSLPrimitive(
                name="rotate90",
                operation=lambda grid: np.rot90(grid),
                parameters=[],
                description="Rotate grid 90 degrees clockwise",
                complexity=0.2
            ),
            DSLPrimitive(
                name="mirror",
                operation=lambda grid: np.fliplr(grid),
                parameters=[],
                description="Mirror the grid horizontally",
                complexity=0.2
            ),
            DSLPrimitive(
                name="fill",
                operation=lambda grid, value: np.full_like(grid, value),
                parameters=["value"],
                description="Fill entire grid with a value",
                complexity=0.1
            ),
            DSLPrimitive(
                name="get_border",
                operation=lambda grid: np.pad(grid, ((1,1), (1,1)), 'constant', constant_values=0)[1:-1,1:-1],
                parameters=[],
                description="Get the border cells of the grid",
                complexity=0.3
            ),
            DSLPrimitive(
                name="get_interior",
                operation=lambda grid: grid[1:-1,1:-1],
                parameters=[],
                description="Get the interior cells of the grid",
                complexity=0.3
            ),
            DSLPrimitive(
                name="flood_fill",
                operation=lambda grid, x, y, value: self._flood_fill(grid.copy(), x, y, grid[x,y], value),
                parameters=["x", "y", "value"],
                description="Fill connected region starting at (x,y) with value",
                complexity=0.4
            ),
            DSLPrimitive(
                name="replace_color",
                operation=lambda grid, old_val, new_val: np.where(grid == old_val, new_val, grid),
                parameters=["old_val", "new_val"],
                description="Replace all occurrences of old_val with new_val",
                complexity=0.2
            ),
            DSLPrimitive(
                name="extract_shape",
                operation=lambda grid, value: np.where(grid == value, 1, 0),
                parameters=["value"],
                description="Extract binary mask of cells with given value",
                complexity=0.3
            ),
            DSLPrimitive(
                name="apply_mask",
                operation=lambda grid, mask, value: np.where(mask == 1, value, grid),
                parameters=["mask", "value"],
                description="Apply value to grid where mask is 1",
                complexity=0.3
            ),
            DSLPrimitive(
                name="find_pattern",
                operation=lambda grid, pattern_value: find_pattern(grid, pattern_value),
                parameters=["pattern_value"],
                description="Find and label contiguous regions of a specific value",
                complexity=0.4
            ),
            DSLPrimitive(
                name="get_pattern_bounds",
                operation=lambda grid, pattern_value: get_pattern_bounds(grid, pattern_value),
                parameters=["pattern_value"],
                description="Get bounding box coordinates of a pattern",
                complexity=0.3
            ),
            DSLPrimitive(
                name="expand_pattern",
                operation=lambda grid, pattern_value, expansion: expand_pattern(grid, pattern_value, expansion),
                parameters=["pattern_value", "expansion"],
                description="Expand a pattern outward by N cells",
                complexity=0.5
            )
        ]
        
        # Add all primitives
        for primitive in primitives:
            self.add_primitive(primitive)
            
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

    def add_primitive(self, primitive: DSLPrimitive):
        """Add a new primitive to the library"""
        if primitive.name not in self.primitives:
            self.primitives[primitive.name] = primitive

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

    def learn_new_primitives(self, task_features: Dict[str, Any], 
                           performance_data: Dict[str, Any]) -> List[DSLPrimitive]:
        """Learn new primitives based on task patterns and performance data"""
        if not self.llm:
            return []
            
        # Create prompt for primitive suggestion
        prompt = (
            f"Based on the following task features and performance data, "
            f"suggest new primitive operations that could help solve similar tasks:\n\n"
            f"Task Features:\n{json.dumps(task_features, indent=2)}\n\n"
            f"Performance Data:\n{json.dumps(performance_data, indent=2)}\n\n"
            f"Current Primitives:\n"
            + "\n".join(f"- {name}: {p.description}" for name, p in self.primitives.items())
            + "\n\n"
            f"Return suggestions in this JSON format:\n"
            f"{{\n"
            f"  'primitives': [\n"
            f"    {{\n"
            f"      'name': 'primitive_name',\n"
            f"      'description': 'what it does',\n"
            f"      'parameters': ['param1', 'param2'],\n"
            f"      'implementation_hint': 'how to implement it',\n"
            f"      'complexity': 0.0-1.0\n"
            f"    }}\n"
            f"  ]\n"
            f"}}"
        )
        
        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[
                {"role": "system", "content": "You are an expert at designing primitive operations for visual pattern transformation. Focus on operations that are reusable across different tasks."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            new_primitives = []
            
            for suggestion in result['primitives']:
                # Generate implementation from hint
                implementation = self._generate_implementation(suggestion)
                if implementation:
                    primitive = DSLPrimitive(
                        name=suggestion['name'],
                        operation=implementation,
                        parameters=suggestion['parameters'],
                        description=suggestion['description'],
                        complexity=suggestion['complexity']
                    )
                    self.add_primitive(primitive)
                    new_primitives.append(primitive)
                    
                    # Record in history
                    self.primitive_history.append({
                        'primitive': primitive,
                        'task_features': task_features,
                        'performance_data': performance_data,
                        'timestamp': time.time()
                    })
                    
            return new_primitives
        except Exception as e:
            print(f"Error learning new primitives: {e}")
            return []
            
    def _generate_implementation(self, suggestion: Dict[str, Any]) -> Optional[Callable]:
        """Generate implementation code from suggestion"""
        if not self.llm:
            return None
            
        prompt = (
            f"Generate a Python implementation for this primitive operation:\n\n"
            f"Name: {suggestion['name']}\n"
            f"Description: {suggestion['description']}\n"
            f"Parameters: {suggestion['parameters']}\n"
            f"Implementation Hint: {suggestion['implementation_hint']}\n\n"
            f"Return only the function definition using numpy operations.\n"
            f"The function should take a grid (numpy array) as first parameter.\n"
            f"Example:\n"
            f"def operation_name(grid, param1, param2):\n"
            f"    # Implementation\n"
            f"    return result"
        )
        
        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[
                {"role": "system", "content": "You are an expert Python programmer specializing in numpy operations for grid transformations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        try:
            code = response.choices[0].message.content.strip()
            # Create function namespace with numpy
            namespace = {'np': np}
            # Execute the function definition
            exec(code, namespace)
            # Get the function from namespace
            func_name = code.split('def ')[1].split('(')[0]
            return namespace[func_name]
        except Exception as e:
            print(f"Error generating implementation: {e}")
            return None
            
    def evaluate_primitive(self, primitive: DSLPrimitive, 
                         success_rate: float, usage_count: int):
        """Update primitive performance metrics"""
        if primitive.name not in self.primitives:
            return
            
        p = self.primitives[primitive.name]
        p.usage_count += usage_count
        # Update success rate with exponential moving average
        alpha = 0.3  # Learning rate
        p.success_rate = (1 - alpha) * p.success_rate + alpha * success_rate
        
        # Remove primitive if it's performing poorly after sufficient usage
        if p.usage_count > 10 and p.success_rate < 0.2:
            del self.primitives[primitive.name]

    def analyze_primitive_effectiveness(self, task_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze which primitives are most effective for a given task context"""
        if not self.llm:
            return []
            
        prompt = f"""Analyze which primitives would be most effective for this task:

Task Context:
{json.dumps(task_context, indent=2)}

Available Primitives:
{json.dumps([{
    'name': p.name,
    'description': p.description,
    'success_rate': p.success_rate,
    'usage_count': p.usage_count
} for p in self.primitives.values()], indent=2)}

Return analysis in JSON format:
{{
    "recommended_primitives": [
        {{
            "name": "primitive_name",
            "relevance_score": 0.0-1.0,
            "usage_suggestion": "how to use it effectively",
            "potential_combinations": ["other primitives to combine with"]
        }}
    ]
}}"""
        
        response = self.llm.get_completion(prompt)
        try:
            result = json.loads(response)
            return result.get('recommended_primitives', [])
        except (json.JSONDecodeError, KeyError):
            return []
            
    def evolve_primitive(self, primitive_name: str, 
                        performance_data: Dict[str, Any]) -> Optional[DSLPrimitive]:
        """Evolve a primitive based on its performance data"""
        if primitive_name not in self.primitives or not self.llm:
            return None
            
        primitive = self.primitives[primitive_name]
        
        prompt = f"""This primitive needs to be evolved based on its performance:

Current Primitive:
{{
    "name": "{primitive.name}",
    "description": "{primitive.description}",
    "success_rate": {primitive.success_rate},
    "usage_count": {primitive.usage_count}
}}

Performance Data:
{json.dumps(performance_data, indent=2)}

Suggest improvements in JSON format:
{{
    "evolved_primitive": {{
        "name": "new_name",
        "description": "improved description",
        "implementation_suggestion": "how to improve the implementation",
        "new_parameters": ["list of parameters"],
        "evolution_rationale": "why these changes help"
    }}
}}"""
        
        response = self.llm.get_completion(prompt)
        try:
            result = json.loads(response)
            suggestion = result['evolved_primitive']
            
            # Generate new implementation
            implementation = self._generate_implementation(suggestion)
            if implementation:
                new_primitive = DSLPrimitive(
                    name=suggestion['name'],
                    operation=implementation,
                    parameters=suggestion['new_parameters'],
                    description=suggestion['description'],
                    complexity=primitive.complexity,
                    usage_count=0,
                    success_rate=0.0
                )
                
                # Record evolution
                self.primitive_history.append({
                    'timestamp': time.time(),
                    'original': primitive_name,
                    'evolved': suggestion['name'],
                    'rationale': suggestion['evolution_rationale'],
                    'performance_data': performance_data
                })
                
                return new_primitive
                
        except (json.JSONDecodeError, KeyError):
            return None
            
    def get_primitive_history(self, primitive_name: str) -> List[Dict[str, Any]]:
        """Get the evolution history of a primitive"""
        return [
            entry for entry in self.primitive_history
            if entry['original'] == primitive_name or entry['evolved'] == primitive_name
        ]
        
    def suggest_primitive_combination(self, primitives: List[str],
                                   task_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Suggest how to combine primitives effectively"""
        if not self.llm:
            return None
            
        primitive_details = []
        for name in primitives:
            if name in self.primitives:
                p = self.primitives[name]
                primitive_details.append({
                    'name': p.name,
                    'description': p.description,
                    'parameters': p.parameters
                })
                
        if not primitive_details:
            return None
            
        prompt = f"""Suggest how to combine these primitives effectively:

Primitives to Combine:
{json.dumps(primitive_details, indent=2)}

Task Context:
{json.dumps(task_context, indent=2)}

Suggest combination in JSON format:
{{
    "combined_primitive": {{
        "name": "descriptive_name",
        "description": "what it does",
        "implementation_logic": "how to combine the primitives",
        "parameters": ["required parameters"],
        "execution_order": ["ordered list of primitive names"],
        "parameter_mapping": {{
            "primitive_name": {{"param": "value"}}
        }}
    }}
}}"""
        
        response = self.llm.get_completion(prompt)
        try:
            result = json.loads(response)
            return result.get('combined_primitive')
        except (json.JSONDecodeError, KeyError):
            return None
            
def find_pattern(grid, pattern_value):
    """Find contiguous regions of a specific value"""
    mask = grid == pattern_value
    labeled, num_features = np.zeros_like(grid), 0
    if mask.any():
        from scipy.ndimage import label
        labeled, num_features = label(mask)
    return labeled, num_features

def get_pattern_bounds(grid, pattern_value):
    """Get bounding box of pattern with specific value"""
    mask = grid == pattern_value
    if not mask.any():
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (rmin, rmax+1, cmin, cmax+1)

def expand_pattern(grid, pattern_value, expansion=1):
    """Expand a pattern outward by N cells"""
    from scipy.ndimage import binary_dilation
    mask = grid == pattern_value
    kernel = np.ones((3,3))
    expanded = mask
    for _ in range(expansion):
        expanded = binary_dilation(expanded, kernel)
    result = grid.copy()
    result[expanded & (grid != pattern_value)] = pattern_value
    return result
