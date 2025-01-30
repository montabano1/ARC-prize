from typing import List, Dict, Any, Optional
import numpy as np

class Curriculum:
    """Manages learning progression through tasks"""
    def __init__(self):
        self.current_level = 0.0
        self.performance_history: List[float] = []
        self.focus_areas: List[str] = []
        
    def update_progress(self, success_rate: float):
        """Update curriculum based on task performance"""
        self.performance_history.append(success_rate)
        
        # Calculate moving average of recent performance
        window_size = min(5, len(self.performance_history))
        recent_performance = np.mean(self.performance_history[-window_size:])
        
        # Adjust current level based on performance
        if recent_performance > 0.8:
            # Doing well, increase difficulty
            self.current_level = min(1.0, self.current_level + 0.1)
        elif recent_performance < 0.5:
            # Struggling, decrease difficulty
            self.current_level = max(0.0, self.current_level - 0.1)
            
        # Update focus areas based on performance
        self._update_focus_areas(recent_performance)
        
    def _update_focus_areas(self, recent_performance: float):
        """Update areas that need focus based on performance"""
        self.focus_areas = []
        
        if recent_performance < 0.6:
            self.focus_areas.append("Pattern Recognition")
        if recent_performance < 0.5:
            self.focus_areas.append("Basic Transformations")
        if recent_performance < 0.4:
            self.focus_areas.append("Grid Operations")
            
    def get_next_task_requirements(self) -> Dict[str, Any]:
        """Get requirements for selecting the next task"""
        return {
            'max_difficulty': self.current_level + 0.2,
            'min_difficulty': max(0.0, self.current_level - 0.1),
            'focus_areas': self.focus_areas.copy(),
            'recent_performance': np.mean(self.performance_history[-5:]) 
                if self.performance_history else 0.0
        }
