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
    def __init__(self, primitive_library: DynamicPrimitiveLibrary):
        self.library = primitive_library
        self.program_cache: Dict[str, DSLProgram] = {}
        
    def synthesize_program(self, task_features: TaskFeatures) -> Optional[DSLProgram]:
        """Synthesize a DSL program based on identified patterns"""
        patterns = task_features.identified_patterns
        
        # Check cache first
        cache_key = self._get_cache_key(patterns)
        if cache_key in self.program_cache:
            return self.program_cache[cache_key]
            
        # Get concepts from task features
        concepts = task_features.extracted_concepts
        if not concepts:
            return None
            
        # First try using unified strategy concept
        strategy_concept = next((c for c in concepts if c.name.startswith('strategy_')), None)
        if strategy_concept and strategy_concept.dsl_template:
            steps = self._instantiate_template(strategy_concept.dsl_template)
            if steps:
                program = DSLProgram(
                    steps=steps,
                    complexity=self._calculate_complexity(steps),
                    description=self._generate_description(steps)
                )
                self.program_cache[cache_key] = program
                return program
                
        # If no strategy concept or it failed, try combining other concepts
        steps = []
        for concept in concepts:
            if concept.dsl_template:
                concept_steps = self._instantiate_template(concept.dsl_template)
                if concept_steps:
                    steps.extend(concept_steps)
                    
        if not steps:
            return None
            
        # Create program
        program = DSLProgram(
            steps=steps,
            complexity=self._calculate_complexity(steps),
            description=self._generate_description(steps)
        )
        
        # Cache program
        self.program_cache[cache_key] = program
        return program
        
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
        
    def _calculate_complexity(self, steps: List[Dict[str, Any]]) -> float:
        """Calculate program complexity"""
        return sum(
            self.library.get_primitive(step['primitive']).complexity
            for step in steps
        )
        
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
        return result
