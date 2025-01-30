from typing import Dict, List, Any, Tuple
import numpy as np
from src.llm.llm_interface import LLMInterface
from src.concept.concept_formation import ConceptFormation, Concept
from dataclasses import dataclass

@dataclass
class TaskFeatures:
    grid_size: Tuple[int, int]
    unique_colors: int
    pattern_complexity: float
    overall_difficulty: float
    identified_patterns: Dict[str, List[Dict[str, Any]]]
    confidence_scores: Dict[str, float]
    extracted_concepts: List[Concept]

class TaskAnalyzer:
    def __init__(self, llm: LLMInterface, concept_formation: ConceptFormation):
        self.llm = llm
        self.concept_formation = concept_formation
        
    def analyze_task(self, task_data: Dict[str, Any]) -> TaskFeatures:
        """Analyze an ARC task and return its features."""
        
        if not task_data.get('train'):
            raise ValueError("No training examples provided")
            
        # Use LLM to analyze all training examples together
        patterns = self.llm.analyze_pattern(task_data)
        
        # Get the first example for basic metrics
        first_example = task_data['train'][0]
        input_grid = np.array(first_example['input'])
        output_grid = np.array(first_example['output'])
        
        # Calculate average complexity across all examples
        total_complexity = 0
        for example in task_data['train']:
            example_input = np.array(example['input'])
            example_output = np.array(example['output'])
            total_complexity += self._calculate_pattern_complexity(example_input, example_output)
        avg_complexity = total_complexity / len(task_data['train'])
        
        # Calculate confidence scores
        confidence_by_category = {
            category: sum(p.get('confidence', 0.8) for p in patterns.get(category, []) if isinstance(p, dict)) / max(1, len([p for p in patterns.get(category, []) if isinstance(p, dict)]))
            for category in patterns.keys() if patterns.get(category)
        }
        
        # Extract concepts from patterns
        concepts = self.concept_formation.extract_concepts(TaskFeatures(
            grid_size=input_grid.shape,
            unique_colors=len(np.unique(input_grid)),
            pattern_complexity=avg_complexity,
            overall_difficulty=self._calculate_overall_difficulty(
                input_grid.shape,
                len(np.unique(input_grid)),
                avg_complexity,
                len(task_data['train'])
            ),
            identified_patterns=patterns,
            confidence_scores=confidence_by_category,
            extracted_concepts=[]  # Temporary empty list
        ))
        
        return TaskFeatures(
            grid_size=input_grid.shape,
            unique_colors=len(np.unique(input_grid)),
            pattern_complexity=avg_complexity,
            overall_difficulty=self._calculate_overall_difficulty(
                input_grid.shape,
                len(np.unique(input_grid)),
                avg_complexity,
                len(task_data['train'])
            ),
            identified_patterns=patterns,
            confidence_scores=confidence_by_category,
            extracted_concepts=concepts
        )
        
    def _calculate_pattern_complexity(self, input_grid: np.ndarray, 
                                   output_grid: np.ndarray) -> float:
        """Calculate complexity of pattern transformation."""
        # Basic complexity based on grid properties
        size_complexity = np.prod(input_grid.shape) / 25  # Normalize by 5x5
        color_complexity = len(np.unique(input_grid)) / 3  # Normalize by 3 colors
        
        # Transformation complexity
        diff_ratio = np.sum(input_grid != output_grid) / input_grid.size
        transform_complexity = diff_ratio * 2  # Scale up differences
        
        # Combine metrics
        return (0.3 * size_complexity + 
                0.3 * color_complexity + 
                0.4 * transform_complexity)
        
    def _calculate_overall_difficulty(self, grid_size: Tuple[int, int], 
                                   unique_colors: int, complexity: float,
                                   num_examples: int) -> float:
        """Calculate overall task difficulty."""
        # Size factor
        size_factor = np.prod(grid_size) / 25  # Normalize by 5x5
        
        # Color factor
        color_factor = unique_colors / 3  # Normalize by 3 colors
        
        # Example factor (fewer examples = harder)
        example_factor = 1 + (1 / num_examples)
        
        # Combine factors with complexity
        difficulty = (
            0.3 * size_factor +
            0.2 * color_factor +
            0.3 * complexity +
            0.2 * example_factor
        )
        
        return min(1.0, difficulty)  # Cap at 1.0
