import logging
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
from src.storage.learning_state import LearningStateManager
from src.llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class TaskAssessor:
    """Assesses task complexity and required concepts"""
    
    def __init__(self, llm: LLMInterface, state_manager: LearningStateManager):
        self.llm = llm
        self.state_manager = state_manager
        self.complexity_factors = {
            "grid_size": 0.2,
            "object_count": 0.15,
            "transformation_count": 0.25,
            "pattern_complexity": 0.25,
            "concept_dependency": 0.15
        }
        
    async def analyze_complexity(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task complexity across multiple dimensions"""
        # Extract basic metrics
        input_grid = task.get("input", [[]])
        output_grid = task.get("output", [[]])
        
        metrics = {
            "grid_size": self._calculate_grid_complexity(input_grid, output_grid),
            "object_count": self._count_distinct_objects(input_grid),
            "transformation_count": await self._estimate_transformations(task),
            "pattern_complexity": await self._analyze_pattern_complexity(task),
            "concept_dependency": await self._analyze_concept_dependencies(task)
        }
        
        # Calculate weighted complexity score
        complexity_score = sum(
            metrics[factor] * weight 
            for factor, weight in self.complexity_factors.items()
        )
        
        # Record analysis
        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task.get("id", "unknown"),
            "complexity_score": complexity_score,
            "metrics": metrics
        }
        
        self.state_manager.record_pattern({
            "type": "complexity_analysis",
            "data": analysis_result
        })
        
        return analysis_result
        
    async def identify_required_concepts(self, task: Dict[str, Any]) -> List[str]:
        """Identify concepts needed to solve the task"""
        # Get pattern analysis from LLM
        pattern_analysis = await self.llm.analyze_pattern(task)
        
        # Extract concepts from patterns
        concepts = []
        for pattern_type in ["shape_patterns", "color_patterns", "spatial_patterns"]:
            patterns = pattern_analysis.get("analysis", {}).get(pattern_type, [])
            for pattern in patterns:
                # Use LLM to map pattern to concept
                concept_prompt = f"""
                Given this pattern: {pattern}
                What fundamental concept does this pattern represent?
                Consider concepts like:
                - Symmetry
                - Rotation
                - Translation
                - Color transformation
                - Shape transformation
                - Pattern repetition
                """
                
                response = await self.llm.explain_strategy({"prompt": concept_prompt})
                concepts.append(response.strip())
                
        # Remove duplicates while preserving order
        unique_concepts = list(dict.fromkeys(concepts))
        
        # Record identified concepts
        self.state_manager.record_pattern({
            "type": "concept_identification",
            "data": {
                "task_id": task.get("id", "unknown"),
                "identified_concepts": unique_concepts
            }
        })
        
        return unique_concepts
        
    async def estimate_difficulty(self, task: Dict[str, Any]) -> float:
        """Estimate overall task difficulty"""
        # Get complexity analysis
        complexity = await self.analyze_complexity(task)
        
        # Get required concepts
        required_concepts = await self.identify_required_concepts(task)
        
        # Calculate concept difficulty
        concept_difficulties = []
        for concept in required_concepts:
            # Get historical performance for concept
            patterns = self.state_manager.get_relevant_patterns({
                "concept": concept
            })
            
            if patterns:
                success_rates = [
                    p.get("success_rate", 0) 
                    for p in patterns
                ]
                avg_success_rate = np.mean(success_rates)
                concept_difficulties.append(1 - avg_success_rate)
            else:
                # New concept, assume moderate difficulty
                concept_difficulties.append(0.5)
                
        # Combine complexity and concept difficulty
        base_difficulty = complexity["complexity_score"]
        concept_difficulty = np.mean(concept_difficulties) if concept_difficulties else 0.5
        
        difficulty_score = 0.6 * base_difficulty + 0.4 * concept_difficulty
        
        # Record difficulty estimation
        self.state_manager.record_pattern({
            "type": "difficulty_estimation",
            "data": {
                "task_id": task.get("id", "unknown"),
                "difficulty_score": difficulty_score,
                "base_difficulty": base_difficulty,
                "concept_difficulty": concept_difficulty
            }
        })
        
        return difficulty_score
        
    def _calculate_grid_complexity(self, input_grid: List[List[int]], 
                                 output_grid: List[List[int]]) -> float:
        """Calculate complexity based on grid sizes"""
        input_size = len(input_grid) * len(input_grid[0]) if input_grid else 0
        output_size = len(output_grid) * len(output_grid[0]) if output_grid else 0
        
        # Normalize to 0-1 range (assuming max size of 30x30)
        max_size = 900  # 30x30
        return min(1.0, (input_size + output_size) / (2 * max_size))
        
    def _count_distinct_objects(self, grid: List[List[int]]) -> float:
        """Count distinct non-zero values in grid"""
        unique_values = set()
        for row in grid:
            unique_values.update(set(v for v in row if v != 0))
            
        # Normalize to 0-1 range (assuming max of 10 distinct objects)
        return min(1.0, len(unique_values) / 10)
        
    async def _estimate_transformations(self, task: Dict[str, Any]) -> float:
        """Estimate number of transformations needed"""
        # Get transformation rules from LLM analysis
        analysis = await self.llm.analyze_pattern(task)
        rules = analysis.get("analysis", {}).get("transformation_rules", [])
        
        # Normalize to 0-1 range (assuming max of 5 transformations)
        return min(1.0, len(rules) / 5)
        
    async def _analyze_pattern_complexity(self, task: Dict[str, Any]) -> float:
        """Analyze complexity of patterns in the task"""
        # Get pattern analysis from LLM
        analysis = await self.llm.analyze_pattern(task)
        
        # Count total patterns
        pattern_types = ["shape_patterns", "color_patterns", "spatial_patterns"]
        total_patterns = sum(
            len(analysis.get("analysis", {}).get(pt, []))
            for pt in pattern_types
        )
        
        # Normalize to 0-1 range (assuming max of 10 patterns)
        return min(1.0, total_patterns / 10)
        
    async def _analyze_concept_dependencies(self, task: Dict[str, Any]) -> float:
        """Analyze concept dependencies in the task"""
        # Get required concepts
        concepts = await self.identify_required_concepts(task)
        
        # Calculate dependency complexity based on number of concepts
        # and their relationships
        dependency_count = len(concepts)
        
        # Check for concept relationships
        relationship_count = 0
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                # Check if concepts are often used together
                patterns = self.state_manager.get_relevant_patterns({
                    "concepts": {"$all": [concept1, concept2]}
                })
                if patterns:
                    relationship_count += 1
                    
        # Normalize to 0-1 range
        # Max complexity would be 5 concepts with 10 relationships
        concept_factor = min(1.0, dependency_count / 5)
        relationship_factor = min(1.0, relationship_count / 10)
        
        return 0.7 * concept_factor + 0.3 * relationship_factor
