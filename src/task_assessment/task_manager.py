from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from src.task_assessment.task_analyzer import TaskAnalyzer
from src.dsl.synthesizer import DSLSynthesizer
from src.concept.concept_formation import ConceptFormation
from src.curriculum.curriculum import Curriculum
from src.optimization.performance_optimizer import PerformanceOptimizer

@dataclass
class TaskResult:
    """Results from processing a task"""
    grid_size: tuple
    unique_colors: int
    pattern_complexity: float
    overall_difficulty: float
    identified_patterns: Dict[str, List[Dict[str, Any]]]
    confidence_scores: Dict[str, float]
    dsl_program: Optional['DSLProgram'] = None
    learned_concepts: List['Concept'] = None
    success_rate: float = 0.0

class TaskManager:
    """Manages the processing and learning from ARC tasks"""
    def __init__(self, 
                 task_analyzer: TaskAnalyzer,
                 dsl_synthesizer: DSLSynthesizer,
                 concept_formation: ConceptFormation,
                 curriculum: Curriculum,
                 performance_optimizer: PerformanceOptimizer):
        self.task_analyzer = task_analyzer
        self.dsl_synthesizer = dsl_synthesizer
        self.concept_formation = concept_formation
        self.curriculum = curriculum
        self.performance_optimizer = performance_optimizer
        
    def process_task(self, task_data: Dict[str, Any]) -> TaskResult:
        """Process a task and learn from it"""
        # Analyze task patterns
        task_features = self.task_analyzer.analyze_task(task_data)
        
        # Extract concepts
        concepts = self.concept_formation.extract_concepts(task_features)
        
        # Generate DSL program
        program = self.dsl_synthesizer.synthesize_program(task_features)
        
        # Test program on training examples
        success_rate = 0.0
        if program:
            success = 0
            total = 0
            for example in task_data['train']:
                input_grid = np.array(example['input'])
                expected = np.array(example['output'])
                actual = self.dsl_synthesizer.execute_program(program, input_grid)
                
                if np.array_equal(actual, expected):
                    success += 1
                total += 1
                
            success_rate = success / total if total > 0 else 0.0
            
            # Update concept performance
            for concept in concepts:
                self.concept_formation.update_concept_performance(
                    concept.name, 
                    success_rate > 0.8
                )
                
            # Update curriculum based on performance
            self.curriculum.update_progress(success_rate)
            
            # Optimize performance if needed
            if success_rate < 0.8:
                program = self.performance_optimizer.optimize_program(
                    program, task_data, success_rate
                )
        
        return TaskResult(
            grid_size=task_features.grid_size,
            unique_colors=task_features.unique_colors,
            pattern_complexity=task_features.pattern_complexity,
            overall_difficulty=task_features.overall_difficulty,
            identified_patterns=task_features.identified_patterns,
            confidence_scores=task_features.confidence_scores,
            dsl_program=program,
            learned_concepts=concepts,
            success_rate=success_rate
        )
