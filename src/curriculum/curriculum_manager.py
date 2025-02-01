from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
from src.task_assessment.task_analyzer import TaskFeatures
from src.llm.llm_interface import LLMInterface
from src.storage.learning_state import LearningStateManager

@dataclass
class Task:
    id: str
    input_grids: List[np.ndarray]
    output_grids: List[np.ndarray]
    features: TaskFeatures
    difficulty: float
    prerequisites: List[str]
    skills_required: List[str]
    skills_taught: List[str]

@dataclass
class LearningPath:
    tasks: List[Task]
    estimated_completion_time: float
    skill_progression: Dict[str, float]
    confidence_threshold: float

class CurriculumManager:
    """Manages the learning curriculum and task selection."""
    
    def __init__(self, llm: LLMInterface, state_manager: LearningStateManager,
                 initial_difficulty: float = 0.3,
                 confidence_threshold: float = 0.8):
        """Initialize curriculum manager."""
        self.llm = llm
        self.state_manager = state_manager
        self.tasks = {}
        self.current_difficulty = initial_difficulty
        self.confidence_threshold = confidence_threshold
        self.learned_skills = {}
        self.task_history = []
        
    def add_task(self, task: Task):
        """Add a new task to the curriculum."""
        self.tasks[task.id] = task
        
    def get_next_task(self) -> Optional[Task]:
        """Get the next appropriate task based on current progress."""
        available_tasks = self._filter_available_tasks()
        if not available_tasks:
            return None
            
        # Sort tasks by how well they match current learning needs
        scored_tasks = [
            (task, self._calculate_task_score(task))
            for task in available_tasks
        ]
        
        # Return the highest scoring task
        best_task = max(scored_tasks, key=lambda x: x[1])[0]
        return best_task
        
    def update_progress(self, task_id: str, performance_score: float):
        """Update learning progress after completing a task."""
        if task_id not in self.tasks:
            raise ValueError(f"Unknown task: {task_id}")
            
        task = self.tasks[task_id]
        self.task_history.append((task_id, performance_score))
        
        # Update learned skills
        if performance_score >= self.confidence_threshold:
            for skill in task.skills_taught:
                current_level = self.learned_skills.get(skill, 0.0)
                self.learned_skills[skill] = min(1.0, current_level + 0.2)
                
        # Adjust difficulty based on performance
        self._adjust_difficulty(performance_score)
        
    def generate_learning_path(self, target_skills: List[str]) -> LearningPath:
        """Generate a learning path to achieve target skills."""
        required_tasks = []
        current_skills = set(self.learned_skills.keys())
        target_skills_set = set(target_skills)
        
        while current_skills < target_skills_set:
            # Find tasks that teach missing skills
            missing_skills = target_skills_set - current_skills
            candidate_tasks = [
                task for task in self.tasks.values()
                if any(skill in missing_skills for skill in task.skills_taught)
                and all(prereq in current_skills for prereq in task.prerequisites)
            ]
            
            if not candidate_tasks:
                break
                
            # Choose the most appropriate task
            best_task = max(candidate_tasks,
                          key=lambda t: self._calculate_task_value(t, missing_skills))
            required_tasks.append(best_task)
            
            # Update current skills
            current_skills.update(best_task.skills_taught)
            
        return LearningPath(
            tasks=required_tasks,
            estimated_completion_time=self._estimate_completion_time(required_tasks),
            skill_progression=self._calculate_skill_progression(required_tasks),
            confidence_threshold=self.confidence_threshold
        )
        
    def _filter_available_tasks(self) -> List[Task]:
        """Filter tasks based on prerequisites and current difficulty."""
        return [
            task for task in self.tasks.values()
            if self._meets_prerequisites(task)
            and abs(task.difficulty - self.current_difficulty) <= 0.2
        ]
        
    def _meets_prerequisites(self, task: Task) -> bool:
        """Check if all prerequisites for a task are met."""
        return all(
            prereq in self.learned_skills
            and self.learned_skills[prereq] >= self.confidence_threshold
            for prereq in task.prerequisites
        )
        
    def _calculate_task_score(self, task: Task) -> float:
        """Calculate how appropriate a task is for current learning state."""
        # Consider multiple factors
        difficulty_match = 1 - abs(task.difficulty - self.current_difficulty)
        skill_relevance = self._calculate_skill_relevance(task)
        novelty = self._calculate_novelty(task)
        
        return 0.4 * difficulty_match + 0.4 * skill_relevance + 0.2 * novelty
        
    def _calculate_skill_relevance(self, task: Task) -> float:
        """Calculate how relevant the task's skills are to current learning needs."""
        if not task.skills_taught:
            return 0.0
            
        relevant_skills = sum(
            1 for skill in task.skills_taught
            if skill not in self.learned_skills
            or self.learned_skills[skill] < self.confidence_threshold
        )
        return relevant_skills / len(task.skills_taught)
        
    def _calculate_novelty(self, task: Task) -> float:
        """Calculate how novel a task is based on task history."""
        if not self.task_history:
            return 1.0
            
        # Check when this task was last attempted
        last_attempt = float('inf')
        for i, (task_id, _) in enumerate(reversed(self.task_history)):
            if task_id == task.id:
                last_attempt = i
                break
                
        return 1.0 / (1.0 + last_attempt)
        
    def _adjust_difficulty(self, performance_score: float):
        """Adjust current difficulty based on performance."""
        if performance_score >= self.confidence_threshold:
            self.current_difficulty = min(1.0, self.current_difficulty + 0.1)
        else:
            self.current_difficulty = max(0.1, self.current_difficulty - 0.1)
            
    def _calculate_task_value(self, task: Task, missing_skills: set) -> float:
        """Calculate the value of a task for learning specific skills."""
        taught_skills = set(task.skills_taught)
        skill_coverage = len(taught_skills & missing_skills) / len(missing_skills)
        difficulty_penalty = abs(task.difficulty - self.current_difficulty)
        return skill_coverage - 0.3 * difficulty_penalty
        
    def _estimate_completion_time(self, tasks: List[Task]) -> float:
        """Estimate time to complete a sequence of tasks."""
        # Simple estimation based on difficulty
        return sum(task.difficulty * 30 for task in tasks)  # minutes
        
    def _calculate_skill_progression(self, tasks: List[Task]) -> Dict[str, float]:
        """Calculate expected skill progression through tasks."""
        progression = self.learned_skills.copy()
        for task in tasks:
            for skill in task.skills_taught:
                current_level = progression.get(skill, 0.0)
                progression[skill] = min(1.0, current_level + 0.2)
        return progression

    async def select_next_task(self, current_accuracy: float) -> Dict[str, Any]:
        """Select next task based on performance."""
        try:
            # Get task history and performance
            history = self.task_history[-5:] if self.task_history else []
            
            # Create selection prompt
            prompt = f"""
            Based on current performance (accuracy: {current_accuracy:.2%}) and task history:
            {history}
            
            Recommend a next task focusing on:
            1. Areas needing improvement
            2. Gradual difficulty progression
            3. Skill reinforcement
            
            Return a JSON object with:
            {{
                "task_type": "string",
                "difficulty": "beginner|intermediate|advanced",
                "focus_areas": ["list", "of", "skills"],
                "description": "task description"
            }}
            """
            
            # Get LLM recommendation
            response = await self.llm.get_completion(prompt)
            
            # Parse response
            try:
                import json
                next_task = json.loads(response)
                return next_task
            except:
                return {
                    "task_type": "practice",
                    "difficulty": "beginner" if current_accuracy < 0.7 else "intermediate",
                    "focus_areas": ["pattern recognition"],
                    "description": "Practice basic pattern recognition"
                }
                
        except Exception as e:
            print(f"Error selecting next task: {str(e)}")
            return {
                "task_type": "review",
                "difficulty": "beginner",
                "focus_areas": ["fundamentals"],
                "description": "Review fundamental concepts"
            }
            
    async def identify_knowledge_gaps(self) -> List[str]:
        """Identify knowledge gaps from learning history."""
        try:
            # Get recent performance data
            history = self.task_history[-10:] if self.task_history else []
            
            # Create analysis prompt
            prompt = f"""
            Analyze this learning history to identify knowledge gaps:
            {history}
            
            Return a JSON array of specific areas needing improvement:
            ["area1", "area2", "area3"]
            """
            
            # Get LLM analysis
            response = await self.llm.get_completion(prompt)
            
            # Parse response
            try:
                import json
                gaps = json.loads(response)
                if isinstance(gaps, list):
                    return gaps
            except:
                pass
                
            return ["pattern recognition", "spatial reasoning"]
            
        except Exception as e:
            print(f"Error identifying knowledge gaps: {str(e)}")
            return ["error: could not analyze gaps"]
            
    def update_task_history(self, task: Dict[str, Any], performance: Dict[str, Any]):
        """Update task history with new task and performance."""
        self.task_history.append({
            "task": task,
            "performance": performance,
            "timestamp": "current_time"
        })
