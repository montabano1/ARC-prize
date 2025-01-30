from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class TaskDifficulty:
    """Difficulty assessment for a task"""
    complexity_score: float
    concept_difficulty: float
    pattern_complexity: float
    overall_score: float

class CurriculumManager:
    """Manages task progression and learning curriculum"""
    def __init__(self):
        self.current_level = 0.0
        self.task_history: List[Dict[str, Any]] = []
        self.performance_history: List[float] = []
        self.difficulty_threshold = 0.2  # Max difficulty increase allowed
        
    def assess_task_difficulty(self, task: Dict[str, Any], 
                             concepts: List[Dict[str, Any]]) -> TaskDifficulty:
        """Assess the difficulty of a task"""
        # Calculate various complexity metrics
        grid_complexity = self._calculate_grid_complexity(task)
        concept_difficulty = self._calculate_concept_difficulty(concepts)
        pattern_complexity = self._calculate_pattern_complexity(task)
        
        # Combine metrics into overall score
        overall_score = (
            0.4 * grid_complexity +
            0.3 * concept_difficulty +
            0.3 * pattern_complexity
        )
        
        return TaskDifficulty(
            complexity_score=grid_complexity,
            concept_difficulty=concept_difficulty,
            pattern_complexity=pattern_complexity,
            overall_score=overall_score
        )
    
    def _calculate_grid_complexity(self, task: Dict[str, Any]) -> float:
        """Calculate complexity based on grid properties"""
        complexity = 0.0
        
        # Get first training example
        if not task.get('train') or not task['train'][0]:
            return complexity
            
        example = task['train'][0]
        input_grid = np.array(example['input'])
        
        # Factor in grid size
        complexity += 0.1 * (input_grid.shape[0] * input_grid.shape[1]) / 25
        
        # Factor in number of unique values
        unique_values = len(np.unique(input_grid))
        complexity += 0.1 * unique_values
        
        # Factor in number of training examples
        complexity += 0.1 * len(task['train'])
        
        return min(complexity, 1.0)
    
    def _calculate_concept_difficulty(self, concepts: List[Dict[str, Any]]) -> float:
        """Calculate difficulty based on required concepts"""
        if not concepts:
            return 0.0
            
        # Average concept complexity
        total_complexity = sum(
            c.get('complexity', 0.5) 
            for c in concepts
        )
        return min(total_complexity / len(concepts), 1.0)
    
    def _calculate_pattern_complexity(self, task: Dict[str, Any]) -> float:
        """Calculate complexity of patterns in the task"""
        complexity = 0.0
        
        # Get first training example
        if not task.get('train') or not task['train'][0]:
            return complexity
            
        example = task['train'][0]
        input_grid = np.array(example['input'])
        output_grid = np.array(example['output'])
        
        # Factor in number of changes between input and output
        changes = np.sum(input_grid != output_grid)
        complexity += 0.2 * (changes / input_grid.size)
        
        return min(complexity, 1.0)
    
    def is_task_appropriate(self, difficulty: TaskDifficulty) -> bool:
        """Check if task difficulty is appropriate for current level"""
        return (
            difficulty.overall_score <= self.current_level + self.difficulty_threshold
            and difficulty.overall_score >= self.current_level - 0.2
        )
    
    def update_curriculum(self, task_result: Dict[str, Any]) -> None:
        """Update curriculum based on task results"""
        # Record task performance
        self.task_history.append(task_result)
        self.performance_history.append(task_result.get('success_rate', 0.0))
        
        # Calculate moving average of recent performance
        window_size = min(10, len(self.performance_history))
        recent_performance = np.mean(self.performance_history[-window_size:])
        
        # Adjust difficulty based on performance
        if recent_performance > 0.8:  # Consistently good performance
            self.current_level = min(1.0, self.current_level + 0.1)
        elif recent_performance < 0.4:  # Consistently poor performance
            self.current_level = max(0.0, self.current_level - 0.1)
    
    def get_next_task_recommendation(self) -> Dict[str, Any]:
        """Get recommendation for next task difficulty"""
        return {
            'target_difficulty': self.current_level,
            'difficulty_range': (
                max(0.0, self.current_level - 0.2),
                min(1.0, self.current_level + self.difficulty_threshold)
            ),
            'focus_areas': self._identify_focus_areas()
        }
    
    def _identify_focus_areas(self) -> List[str]:
        """Identify areas that need more practice"""
        if not self.task_history:
            return ['basic_patterns']
            
        # Analyze recent failures
        recent_tasks = self.task_history[-10:]
        failure_patterns = []
        
        for task in recent_tasks:
            if task.get('success_rate', 0.0) < 0.5:
                failure_patterns.extend(task.get('patterns', []))
                
        # Count pattern frequencies
        pattern_counts = {}
        for pattern in failure_patterns:
            pattern_type = pattern.get('type', 'unknown')
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
        # Return most frequent failure patterns
        sorted_patterns = sorted(
            pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [p[0] for p in sorted_patterns[:3]]
