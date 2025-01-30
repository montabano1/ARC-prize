from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from src.dsl.primitives import DynamicPrimitiveLibrary, DSLPrimitive
from src.task_assessment.task_analyzer import TaskFeatures
from src.concept.concept_formation import Concept

@dataclass
class DSLProgram:
    """A program in the DSL"""
    steps: List[Dict[str, Any]]  # List of {primitive: name, params: dict}
    complexity: float
    description: str
    success_rate: float = 0.0
    
class DSLSynthesizer:
    """Synthesizes DSL programs based on patterns and concepts"""
    def __init__(self, primitive_library: DynamicPrimitiveLibrary, llm):
        self.library = primitive_library
        self.llm = llm
        self.program_cache: Dict[str, DSLProgram] = {}
        self.intermediate_results: Dict[str, Any] = {}
        
    def synthesize_program(self, task_features: TaskFeatures) -> Optional[DSLProgram]:
        """Synthesize a DSL program based on identified patterns"""
        patterns = task_features.identified_patterns
        
        # Check cache first
        cache_key = self._get_cache_key(patterns)
        if cache_key in self.program_cache:
            return self.program_cache[cache_key]
        
        print("\nDEBUG - Available primitives:")
        for name, prim in self.library.primitives.items():
            print(f"- {name}: {prim.description}")
        
        # Generate program from LLM analysis
        program = self._synthesize_from_llm(patterns, task_features)
        if program:
            self.program_cache[cache_key] = program
            print("\nDEBUG - Successfully generated program:")
            print(f"Steps: {program.steps}")
            print(f"Description: {program.description}")
            return program
        else:
            print("\nDEBUG - Failed to generate program. Check LLM response and primitive matching.")
        
        return None

    def _synthesize_from_llm(self, patterns: Dict[str, List[Dict[str, Any]]], task_features: TaskFeatures) -> Optional[DSLProgram]:
        """Use LLM to synthesize program from patterns"""
        # Extract key information from patterns
        pattern_info = {
            'object': patterns.get('object', []),
            'transformation': patterns.get('transformation', []),
            'relationship': patterns.get('relationship', []),
            'abstract': patterns.get('abstract', [])
        }
        
        # Get available primitives
        available_primitives = {
            name: prim.description 
            for name, prim in self.library.primitives.items()
        }
        
        print("\nDEBUG - Requesting program steps with patterns:")
        for k, v in pattern_info.items():
            print(f"{k}: {v}")
        
        # Ask LLM to suggest program steps
        response = self.llm.suggest_program_steps(pattern_info, available_primitives)
        if not response:
            print("\nDEBUG - No program steps suggested by LLM")
            return None
            
        print("\nDEBUG - Processing suggested steps:")
        # Convert LLM suggestions to program steps
        steps = []
        for suggestion in response:
            print(f"\nProcessing step: {suggestion}")
            primitive_name = suggestion.get('primitive')
            if primitive_name in self.library.primitives:
                # Get the primitive's required parameters
                primitive = self.library.primitives[primitive_name]
                params = suggestion.get('params', {})
                
                # Map any mismatched parameter names
                param_mapping = {
                    'value': 'pattern_value',  # Common mismatch
                    'val': 'pattern_value',    # Another possible mismatch
                    'pattern_value': 'value',  # For extract_shape
                    'n': 'expansion'           # For expand_pattern
                }
                
                mapped_params = {}
                for param_name, param_value in params.items():
                    # Use mapped name if it exists and is a required parameter
                    mapped_name = param_mapping.get(param_name, param_name)
                    if mapped_name in primitive.parameters:
                        mapped_params[mapped_name] = param_value
                    elif param_name in primitive.parameters:
                        mapped_params[param_name] = param_value
                
                # Handle special cases for certain primitives
                if 'mask' in primitive.parameters:
                    # For any primitive that needs a mask, try to use the previous step's result
                    if steps:
                        mapped_params['mask'] = f"result_of_{len(steps)-1}"
                
                if 'pattern' in primitive.parameters:
                    # For any primitive that needs a pattern, try to use the previous step's result
                    if steps:
                        mapped_params['pattern'] = f"result_of_{len(steps)-1}"
                
                # Store intermediate results for subsequent steps
                # Any step could potentially be used as input for a future step
                self.intermediate_results[f"result_of_{len(steps)}"] = None
                
                # Verify all required parameters are present
                if all(param in mapped_params for param in primitive.parameters):
                    steps.append({
                        'primitive': primitive_name,
                        'params': mapped_params,
                        'description': suggestion.get('explanation', '')
                    })
                    print(f"Added step using primitive: {primitive_name}")
                    print(f"With parameters: {mapped_params}")
                else:
                    print(f"Missing required parameters for primitive: {primitive_name}")
                    print(f"Required: {primitive.parameters}")
                    print(f"Provided: {list(mapped_params.keys())}")
            else:
                print(f"Primitive not found: {primitive_name}")
        
        if not steps:
            print("\nDEBUG - No valid steps could be created from suggestions")
            return None
            
        return DSLProgram(
            steps=steps,
            complexity=self._calculate_complexity(steps),
            description=f"Program synthesized from pattern analysis:\n" + 
                       f"Strategy: {task_features.unified_strategy}\n" +
                       "\n".join(f"Step {i+1}: {s['description']}" for i, s in enumerate(steps))
        )

    def _calculate_complexity(self, steps: List[Dict[str, Any]]) -> float:
        """Calculate program complexity based on steps"""
        complexity = 0.0
        for step in steps:
            if step['primitive'] in self.library.primitives:
                primitive = self.library.primitives[step['primitive']]
                complexity += primitive.complexity
        return complexity
        
    def _instantiate_template(self, template: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Instantiate a DSL template into concrete steps"""
        steps = []
        for step in template:
            # Verify primitive exists
            if step['primitive'] not in self.library.primitives:
                continue
                
            # Get primitive details
            primitive = self.library.get_primitive(step['primitive'])
            
            # Verify all required parameters are present
            params = step.get('params', {})
            if not all(param in params for param in primitive.parameters):
                continue
                
            steps.append({
                'primitive': step['primitive'],
                'params': params
            })
            
        return steps
        
    def _generate_description(self, steps: List[Dict[str, Any]]) -> str:
        """Generate a description of the program"""
        desc = "Program steps:\n"
        for i, step in enumerate(steps, 1):
            primitive = self.library.get_primitive(step['primitive'])
            params_str = ", ".join(
                f"{k}={v}" for k, v in step.get('params', {}).items()
            )
            desc += f"{i}. {primitive.description}"
            if params_str:
                desc += f" with {params_str}"
            desc += "\n"
        return desc
        
    def _get_cache_key(self, patterns: Dict[str, List[Dict[str, Any]]]) -> str:
        """Generate cache key from patterns"""
        return str(sorted(
            (k, str(sorted(str(p) for p in v)))
            for k, v in patterns.items()
        ))
        
    def execute_program(self, program: DSLProgram, grid: np.ndarray) -> np.ndarray:
        """Execute a DSL program on a grid"""
        result = grid.copy()
        for step in program.steps:
            result = self.library.execute_primitive(
                step['primitive'], result, step.get('params', {})
            )
            if f"result_of_{program.steps.index(step)}" in self.intermediate_results:
                self.intermediate_results[f"result_of_{program.steps.index(step)}"] = result
        return result
