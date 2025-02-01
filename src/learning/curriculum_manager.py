import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
from src.storage.learning_state import LearningStateManager
from src.llm.llm_interface import LLMInterface

logger = logging.getLogger(__name__)

class CurriculumManager:
    """Manages the learning curriculum and progression"""
    
    def __init__(self, llm: LLMInterface, state_manager: LearningStateManager):
        self.llm = llm
        self.state_manager = state_manager
        self.current_level = 0.0
        self.performance_history = []
        self.difficulty_scale = {
            0: "beginner",
            1: "basic",
            2: "intermediate",
            3: "advanced",
            4: "expert",
            5: "master"
        }
        
    async def select_next_task(self, current_performance: float) -> Dict[str, Any]:
        """Select next task based on current learning level and performance"""
        # Update current level based on performance
        self._update_level(current_performance)
        
        # Get relevant strategies for current level
        strategies = self.state_manager.get_effective_strategies({
            "difficulty": self.difficulty_scale[int(self.current_level)],
            "performance_threshold": current_performance
        })
        
        # Get patterns that need reinforcement
        patterns = self.state_manager.get_relevant_patterns({
            "success_rate": {"$lt": 0.8},  # Patterns with < 80% success rate
            "difficulty": {"$lte": self.current_level + 1}  # Not too difficult
        })
        
        # Generate task specification
        task_spec = {
            "difficulty_level": self.current_level,
            "target_concepts": [p["id"] for p in patterns[:3]],  # Focus on up to 3 patterns
            "relevant_strategies": [s["id"] for s in strategies],
            "performance_history": self.performance_history[-5:]  # Last 5 performance records
        }
        
        # Use LLM to generate appropriate task
        prompt = f"""
        Please design a learning task with these specifications:
        Difficulty Level: {self.difficulty_scale[int(self.current_level)]}
        Target Concepts: {task_spec['target_concepts']}
        Recent Performance: {np.mean(self.performance_history[-5:]):.2%}
        
        The task should:
        1. Challenge the learner appropriately
        2. Reinforce concepts with low success rates
        3. Build upon successful strategies
        4. Introduce new concepts gradually
        """
        
        # Get task suggestion from LLM
        response = await self.llm.analyze_pattern({"prompt": prompt})
        
        # Record task selection
        self.state_manager.record_pattern({
            "type": "task_selection",
            "data": {
                "task_spec": task_spec,
                "selected_task": response,
                "current_level": self.current_level
            }
        })
        
        return response
        
    async def adjust_difficulty(self, performance: float) -> None:
        """Adjust task difficulty based on performance"""
        self.performance_history.append(performance)
        
        # Calculate trend
        if len(self.performance_history) >= 3:
            recent_trend = np.mean(self.performance_history[-3:])
            
            # Adjust level based on performance trend
            if recent_trend > 0.85:  # Consistently high performance
                self.current_level = min(5.0, self.current_level + 0.5)
                await self._generate_progress_insight("increase")
            elif recent_trend < 0.6:  # Struggling
                self.current_level = max(0.0, self.current_level - 0.3)
                await self._generate_progress_insight("decrease")
                
        # Record adjustment
        self.state_manager.record_pattern({
            "type": "difficulty_adjustment",
            "data": {
                "new_level": self.current_level,
                "performance_history": self.performance_history[-5:],
                "timestamp": datetime.now().isoformat()
            }
        })
        
    async def identify_knowledge_gaps(self) -> List[str]:
        """Identify areas needing more practice"""
        # Analyze performance patterns
        patterns = self.state_manager.get_relevant_patterns({
            "success_rate": {"$lt": 0.7}  # Patterns with < 70% success rate
        })
        
        # Group patterns by concept
        concept_performance = {}
        for pattern in patterns:
            concept = pattern.get("concept", "unknown")
            if concept not in concept_performance:
                concept_performance[concept] = {
                    "total_attempts": 0,
                    "successful_attempts": 0,
                    "patterns": []
                }
            stats = concept_performance[concept]
            stats["patterns"].append(pattern)
            stats["total_attempts"] += pattern.get("total_attempts", 0)
            stats["successful_attempts"] += pattern.get("successful_attempts", 0)
            
        # Identify gaps
        gaps = []
        for concept, stats in concept_performance.items():
            success_rate = (stats["successful_attempts"] / 
                          max(1, stats["total_attempts"]))
            if success_rate < 0.7:
                gaps.append({
                    "concept": concept,
                    "success_rate": success_rate,
                    "total_attempts": stats["total_attempts"],
                    "pattern_count": len(stats["patterns"])
                })
                
        # Sort gaps by importance (lower success rate and more attempts = more important)
        gaps.sort(key=lambda x: (x["success_rate"], -x["total_attempts"]))
        
        # Generate insights about gaps
        await self._generate_gap_insights(gaps)
        
        return [g["concept"] for g in gaps]
        
    def _update_level(self, performance: float) -> None:
        """Update current level based on performance"""
        # Add performance to history
        self.performance_history.append(performance)
        
        # Calculate moving average
        if len(self.performance_history) >= 5:
            moving_avg = np.mean(self.performance_history[-5:])
            
            # Adjust level based on moving average
            if moving_avg > 0.8:
                self.current_level = min(5.0, self.current_level + 0.2)
            elif moving_avg < 0.6:
                self.current_level = max(0.0, self.current_level - 0.1)
                
    async def _generate_progress_insight(self, direction: str) -> None:
        """Generate insight about progress direction"""
        if direction == "increase":
            message = (f"Performance consistently high ({np.mean(self.performance_history[-3:]):.2%}). " +
                      f"Increasing difficulty to {self.difficulty_scale[int(self.current_level)]}")
        else:
            message = (f"Performance needs improvement ({np.mean(self.performance_history[-3:]):.2%}). " +
                      f"Decreasing difficulty to {self.difficulty_scale[int(self.current_level)]}")
            
        self.state_manager.record_meta_insight(message, importance=0.8)
        
    async def _generate_gap_insights(self, gaps: List[Dict[str, Any]]) -> None:
        """Generate insights about knowledge gaps"""
        if gaps:
            # Generate insight about most significant gap
            worst_gap = gaps[0]
            message = (f"Significant knowledge gap identified in concept '{worst_gap['concept']}' " +
                      f"with only {worst_gap['success_rate']:.2%} success rate over " +
                      f"{worst_gap['total_attempts']} attempts")
            self.state_manager.record_meta_insight(message, importance=0.9)
            
            # Generate insight about overall gaps
            if len(gaps) > 1:
                message = (f"Multiple concepts need attention: " +
                          ", ".join([g['concept'] for g in gaps[:3]]))
                self.state_manager.record_meta_insight(message, importance=0.85)
