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
    unified_strategy: str = ""

class TaskAnalyzer:
    def __init__(self, llm: LLMInterface, concept_formation: ConceptFormation):
        self.llm = llm
        self.concept_formation = concept_formation
        
    async def analyze_task(self, task_data: Dict[str, Any]) -> TaskFeatures:
        """Analyze an ARC task and return its features."""
        
        if not task_data.get('train'):
            raise ValueError("No training examples provided")
            
        # Use LLM to analyze all training examples together
        patterns = await self.llm.analyze_pattern(task_data)
        
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
        concepts = await self.concept_formation.extract_concepts(TaskFeatures(
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
            extracted_concepts=[],  # Temporary empty list
            unified_strategy=""
        ))
        
        # Extract unified strategy from patterns
        unified_strategy = ""
        for pattern_list in patterns.values():
            for pattern in pattern_list:
                if isinstance(pattern, dict) and pattern.get('is_unified_strategy'):
                    unified_strategy = pattern.get('description', '')
                    break
            if unified_strategy:
                break

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
            extracted_concepts=concepts,
            unified_strategy=unified_strategy
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

    async def analyze_complexity(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the complexity of a task based on various factors."""
        try:
            # Get input/output dimensions
            first_example = task_data['train'][0]
            input_grid = np.array(first_example['input'])
            output_grid = np.array(first_example['output'])
            
            # Calculate basic complexity metrics
            grid_size_score = (input_grid.size + output_grid.size) / 100  # Normalize by 100
            unique_values_score = (len(np.unique(input_grid)) + len(np.unique(output_grid))) / 20  # Normalize by 20
            num_examples_score = len(task_data['train']) / 5  # Normalize by 5
            
            # Analyze pattern complexity using LLM
            prompt = f"""
            Analyze the complexity of this ARC task by considering:
            1. The number and complexity of transformations needed
            2. The spatial relationships involved
            3. The pattern recognition difficulty
            4. The logical reasoning steps required

            Training examples:
            {task_data['train']}

            Rate each aspect from 0.0 to 1.0 where 1.0 is most complex.
            Return a JSON object with these scores and a final complexity_score.
            """
            
            response = await self.llm.get_completion(prompt)
            
            try:
                # Extract JSON from response
                import json
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    llm_analysis = json.loads(json_str)
                    
                    # Get scores from LLM analysis
                    transformation_score = llm_analysis.get('transformation_complexity', 0.5)
                    spatial_score = llm_analysis.get('spatial_relationships', 0.5)
                    pattern_score = llm_analysis.get('pattern_recognition', 0.5)
                    logic_score = llm_analysis.get('logical_reasoning', 0.5)
                    
                    # Combine metrics
                    complexity_score = min(1.0, (
                        grid_size_score * 0.1 +
                        unique_values_score * 0.1 +
                        num_examples_score * 0.1 +
                        transformation_score * 0.2 +
                        spatial_score * 0.2 +
                        pattern_score * 0.15 +
                        logic_score * 0.15
                    ))
                    
                    return {
                        'complexity_score': complexity_score,
                        'metrics': {
                            'grid_size': grid_size_score,
                            'unique_values': unique_values_score,
                            'num_examples': num_examples_score,
                            'transformation': transformation_score,
                            'spatial': spatial_score,
                            'pattern': pattern_score,
                            'logic': logic_score
                        }
                    }
            except Exception as e:
                print(f"Error parsing LLM response: {str(e)}")
                
            # Fallback to basic metrics if LLM analysis fails
            complexity_score = min(1.0, (
                grid_size_score * 0.4 +
                unique_values_score * 0.3 +
                num_examples_score * 0.3
            ))
            
            return {
                'complexity_score': complexity_score,
                'metrics': {
                    'grid_size': grid_size_score,
                    'unique_values': unique_values_score,
                    'num_examples': num_examples_score
                }
            }
            
        except Exception as e:
            print(f"Error in analyze_complexity: {str(e)}")
            return {'complexity_score': 0.5}  # Default moderate complexity

    async def identify_required_concepts(self, task_data: Dict[str, Any]) -> List[str]:
        """Identify concepts required to solve this task."""
        try:
            # Get first training example for analysis
            first_example = task_data['train'][0]
            input_grid = np.array(first_example['input'])
            output_grid = np.array(first_example['output'])
            
            # Basic shape and value analysis
            context = {
                'input_shape': input_grid.shape,
                'output_shape': output_grid.shape,
                'input_values': np.unique(input_grid).tolist(),
                'output_values': np.unique(output_grid).tolist(),
                'num_examples': len(task_data['train'])
            }
            
            # Use LLM to identify required concepts
            prompt = f"""
            Analyze this ARC task and identify the key concepts required to solve it.
            Consider concepts like:
            - Pattern recognition
            - Spatial relationships
            - Color/value transformations
            - Shape manipulation
            - Symmetry
            - Sequence recognition
            
            Task context:
            {context}
            
            Training examples:
            {task_data['train']}
            
            List the required concepts, one per line.
            """
            
            concepts = await self.llm.get_completion(prompt)
            return [c.strip() for c in concepts.split('\n') if c.strip()]
            
        except Exception as e:
            print(f"Error identifying required concepts: {str(e)}")
            return ['pattern_recognition']  # Default concept
