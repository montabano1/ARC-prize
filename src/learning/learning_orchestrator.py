from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np
import traceback
import json

from src.llm.llm_interface import LLMInterface
from src.storage.learning_state import LearningStateManager
from src.concept.concept_formation import ConceptFormation
from src.learning.primitive_learning import DynamicPrimitiveLearner
from src.strategy.meta_strategy import MetaStrategyEngine
from src.optimization.performance_optimizer import PerformanceOptimizer
from src.active_learning.learning_components import (
    HypothesisFormer, 
    PatternValidator,
    LearningProgressTracker
)
from src.curriculum.curriculum_manager import CurriculumManager
from src.task_assessment.task_analyzer import TaskAnalyzer

class LearningOrchestrator:
    """Orchestrates the learning process across multiple components."""
    
    def __init__(self):
        """Initialize learning components."""
        from src.llm.llm_interface import LLMInterface
        from src.task_assessment.task_analyzer import TaskAnalyzer
        from src.active_learning.learning_components import (
            HypothesisFormer, PatternValidator, LearningProgressTracker
        )
        from src.storage.learning_state import LearningStateManager
        from src.curriculum.curriculum_manager import CurriculumManager
        from src.optimization.performance_optimizer import PerformanceOptimizer
        from src.concept.concept_formation import ConceptFormation
        from src.learning.primitive_learning import DynamicPrimitiveLearner
        from src.strategy.meta_strategy import MetaStrategyEngine
        
        # Initialize core components first
        self.llm = LLMInterface()
        self.learning_state = LearningStateManager()
        
        # Initialize strategy and learning components
        self.meta_strategy = MetaStrategyEngine(llm=self.llm)
        self.primitive_learner = DynamicPrimitiveLearner(llm=self.llm, state_manager=self.learning_state)
        self.performance_optimizer = PerformanceOptimizer(self.llm)
        self.concept_formation = ConceptFormation(self.llm)
        
        # Initialize active learning components
        self.hypothesis_former = HypothesisFormer(self.llm)
        self.pattern_validator = PatternValidator(self.llm)
        self.progress_tracker = LearningProgressTracker(self.llm)
        
        # Initialize task analysis components
        self.task_analyzer = TaskAnalyzer(self.llm, self.concept_formation)
        self.curriculum_manager = CurriculumManager(self.llm, self.learning_state)
        
        # Initialize history tracking
        self.learning_history = []
        self.feedback_history = []
        self.feedback_requests = []
        
    async def learn_from_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn from a task by analyzing patterns and generating strategies."""
        try:
            print("\nAnalyzing training examples...")
            self._print_examples(task_data)
            
            # Analyze patterns once
            patterns = await self.meta_strategy.analyze_patterns(task_data)
            if not patterns or not patterns[0].get('description'):
                print("No patterns identified")
                return self._create_empty_result()
                
            print("\nIdentified Patterns:")
            for pattern in patterns:
                print(f"- {pattern.get('description', 'No description')}")
                
            # Generate strategy once
            strategy = await self.meta_strategy.generate_strategy(patterns)
            
            print("\nGenerated Strategy:")
            if strategy and strategy.get('description'):
                print(strategy['description'])
            else:
                print("No strategy description available")
                
            # Calculate progress based on pattern confidence
            progress = self._calculate_progress(patterns)
            print("\nLearning Progress:")
            print(f"Overall Progress: {progress:.2f}%")
            
            # Get next recommended task
            next_task = self._get_next_task(progress)
            print(f"\nNext recommended task: {next_task}")
            
            # Generate test example
            test_example = await self.test_generator.generate_test(task_data)
            if test_example:
                print("\nTest Example:")
                print("Input:")
                print(self._format_grid(test_example['input']))
                print("\nExpected Output:")
                print(self._format_grid(test_example['output']))
                
            # Identify areas for practice
            areas = self._identify_practice_areas(patterns)
            print("\nI've identified some areas where we need more practice:", ", ".join(areas))
            
            return {
                "patterns": patterns,
                "strategy": strategy,
                "progress": progress,
                "next_task": next_task,
                "test_example": test_example,
                "practice_areas": areas
            }
            
        except Exception as e:
            print(f"Error in learn_from_task: {str(e)}")
            return self._create_empty_result()
            
    def _calculate_progress(self, patterns: List[Dict[str, Any]]) -> float:
        """Calculate learning progress based on pattern confidence."""
        if not patterns:
            return 0.0
            
        # Average confidence across all patterns
        confidences = [p.get('confidence', 0.0) for p in patterns]
        return sum(confidences) / len(confidences) * 100
        
    def _identify_practice_areas(self, patterns: List[Dict[str, Any]]) -> List[str]:
        """Identify areas that need more practice based on pattern analysis."""
        areas = set()
        
        # Add areas based on pattern confidence
        for pattern in patterns:
            confidence = pattern.get('confidence', 0.0)
            if confidence < 0.8:
                areas.add('pattern recognition')
            if confidence < 0.7:
                areas.add('spatial reasoning')
                
        # Always include these fundamental skills
        areas.add('logical analysis')
        areas.add('problem decomposition')
        
        return list(areas)
        
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create an empty result when learning fails."""
        return {
            "patterns": [],
            "strategy": None,
            "progress": 0.0,
            "next_task": self._get_next_task(0.0),
            "test_example": None,
            "practice_areas": ["pattern recognition", "spatial reasoning"]
        }
        
    def _get_next_task(self, progress: float) -> Dict[str, Any]:
        """Get next recommended task based on progress."""
        if progress < 30:
            return {
                "task_type": "practice",
                "difficulty": "beginner",
                "focus_areas": ["pattern recognition"],
                "description": "Practice basic pattern recognition"
            }
        elif progress < 70:
            return {
                "task_type": "transformation",
                "difficulty": "intermediate",
                "focus_areas": ["pattern recognition", "spatial reasoning"],
                "description": "Apply transformations to more complex grids"
            }
        else:
            return {
                "task_type": "grid transformation",
                "difficulty": "intermediate",
                "focus_areas": ["pattern recognition", "spatial reasoning", "rule deduction", "problem decomposition"],
                "description": "In this task, you are provided with a series of grids representing various abstract patterns. Your goal is to deduce the transformation rules applied between input and output grids and then apply those rules to new input cases. The challenge is designed with increased complexity to gradually progress your skills while reinforcing pattern recognition, spatial reasoning, and the ability to decompose problems into systematic rule-based steps."
            }
            
    def _print_examples(self, task: Dict[str, Any]) -> None:
        """Print training examples."""
        if 'train' in task:
            for i, example in enumerate(task['train'], 1):
                print(f"\nTraining Example {i}:")
                print("Input:")
                print(self._format_grid(example['input']))
                print("\nOutput:")
                print(self._format_grid(example['output']))
                
    def _format_grid(self, grid: List[List[int]]) -> str:
        """Format grid for printing."""
        return '\n'.join([' '.join(map(str, row)) for row in grid])
        
    async def solve_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve an ARC task using active learning."""
        try:
            # First analyze the task
            print("\nAnalyzing training examples...")
            for i, example in enumerate(task_data['train'], 1):
                print(f"\nTraining Example {i}:")
                print("Input:")
                print(np.array(example['input']))
                print("\nOutput:")
                print(np.array(example['output']))
                
            task_features = await self.task_analyzer.analyze_task(task_data)
            
            # Form initial hypothesis
            print("\nForming initial hypothesis...")
            hypothesis = await self.hypothesis_former.form_hypothesis(task_data['train'])
            
            # Validate hypothesis
            print("\nValidating hypothesis...")
            validation = await self.pattern_validator.validate_pattern(
                hypothesis,
                task_data['train']
            )
            
            if not validation['is_valid']:
                print("\nInitial hypothesis was invalid. Refining...")
                # Generate test cases to help refine hypothesis
                test_cases = await self.hypothesis_former.generate_test_cases(hypothesis)
                
                # Update hypothesis based on test results
                test_results = []
                for test_case in test_cases:
                    result = await self._test_hypothesis(hypothesis, test_case)
                    test_results.append(result)
                    if not result['success']:
                        print(f"\nTest case failed: {result['error']}")
                        hypothesis = await self._refine_hypothesis(hypothesis, result)
                
                # Calculate accuracy from test results
                accuracy = sum(1 for r in test_results if r['success']) / len(test_results) if test_results else 0.0
            else:
                accuracy = validation.get('confidence', 0.0)
            
            # Record learning progress
            await self.learning_state.record_task_attempt(
                task_data.get('task_id', 'unknown'),
                validation['is_valid'],
                hypothesis.get('description', '')
            )
            
            # Update curriculum based on performance
            await self.curriculum_manager.adjust_difficulty(accuracy)
            
            # Identify knowledge gaps
            gaps = await self.curriculum_manager.identify_knowledge_gaps()
            if gaps:
                print("\nIdentified knowledge gaps:")
                for gap in gaps:
                    print(f"- {gap}")
                    
            return {
                'hypothesis': hypothesis,
                'validation': validation,
                'task_features': task_features,
                'knowledge_gaps': gaps,
                'accuracy': accuracy
            }
            
        except Exception as e:
            print(f"\nError solving task: {str(e)}")
            return {
                'error': str(e),
                'hypothesis': None,
                'validation': {'is_valid': False, 'confidence': 0.0},
                'task_features': None,
                'knowledge_gaps': [],
                'accuracy': 0.0
            }
            
    async def _test_hypothesis(self, hypothesis: Dict[str, Any], 
                             test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Test a hypothesis against a test case."""
        try:
            # Apply hypothesis transformation
            actual_output = await self._apply_solution_steps(
                test_case['input'],
                hypothesis.get('steps', [])
            )
            
            # Compare with expected output
            success = np.array_equal(
                actual_output,
                np.array(test_case['expected_output'])
            )
            
            return {
                'success': success,
                'error': None if success else 'Output did not match expected'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    async def _refine_hypothesis(self, hypothesis: Dict[str, Any], 
                               test_result: Dict[str, Any]) -> Dict[str, Any]:
        """Refine a hypothesis based on test results."""
        try:
            # Use LLM to analyze failure and suggest refinements
            prompt = f"""
            Original hypothesis:
            {hypothesis['description']}
            
            Test failure:
            {test_result['error']}
            
            Suggest refinements to the hypothesis that would address this failure.
            Return a JSON object with:
            - description: string (refined hypothesis)
            - confidence: float (0-1)
            - changes: array of strings (explaining changes made)
            """
            
            response = await self.llm.get_completion(prompt)
            
            try:
                import json
                refinements = json.loads(response)
                return {
                    'description': refinements['description'],
                    'confidence': refinements['confidence'],
                    'changes': refinements['changes']
                }
            except:
                print("Failed to parse refinements")
                return hypothesis
                
        except Exception as e:
            print(f"Error refining hypothesis: {str(e)}")
            return hypothesis
            
    async def _apply_solution_steps(self, input_grid: List[List[int]], 
                                  steps: List[Dict[str, Any]]) -> np.ndarray:
        """Apply solution steps to input grid."""
        grid = np.array(input_grid)
        
        for step in steps:
            try:
                if step['type'] == 'transform':
                    grid = await self._apply_transformation(grid, step['parameters'])
                elif step['type'] == 'pattern':
                    grid = await self._apply_pattern(grid, step['parameters'])
                else:
                    print(f"Unknown step type: {step['type']}")
            except Exception as e:
                print(f"Error applying step: {str(e)}")
                
        return grid
        
    async def _apply_transformation(self, grid: np.ndarray, 
                                  params: Dict[str, Any]) -> np.ndarray:
        """Apply a transformation to the grid."""
        if params.get('type') == 'rotate':
            return np.rot90(grid, k=params.get('k', 1))
        elif params.get('type') == 'flip':
            return np.flip(grid, axis=params.get('axis', 0))
        elif params.get('type') == 'shift':
            return np.roll(grid, shift=params.get('shift', 1), axis=params.get('axis', 0))
        else:
            raise ValueError(f"Unknown transformation type: {params.get('type')}")
            
    async def _apply_pattern(self, grid: np.ndarray, 
                           params: Dict[str, Any]) -> np.ndarray:
        """Apply a pattern to the grid."""
        if params.get('type') == 'fill':
            grid[params.get('mask')] = params.get('value', 0)
        elif params.get('type') == 'copy':
            source = params.get('source', [0, 0])
            target = params.get('target', [0, 0])
            size = params.get('size', [1, 1])
            grid[target[0]:target[0]+size[0], 
                 target[1]:target[1]+size[1]] = grid[source[0]:source[0]+size[0],
                                                    source[1]:source[1]+size[1]]
        else:
            raise ValueError(f"Unknown pattern type: {params.get('type')}")
            
        return grid

    async def _generate_solution_steps(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate solution steps based on learned patterns and strategies"""
        print("Starting _generate_solution_steps...")
        
        try:
            # Use LLM to analyze input/output and generate steps using basic primitives
            prompt = {
                'task': 'analyze_transformation',
                'input': task_data['input'],
                'output': task_data.get('output'),
                'available_operations': [
                    {
                        'name': 'change_color',
                        'description': 'Change color of cells at specified position',
                        'parameters': ['position', 'color']
                    },
                    {
                        'name': 'move',
                        'description': 'Move a region from source to target position',
                        'parameters': ['source', 'target', 'size']
                    },
                    {
                        'name': 'transform_shape',
                        'description': 'Apply shape transformation (rotate, flip)',
                        'parameters': ['transform', 'position', 'size']
                    }
                ],
                'learned_patterns': self.learning_history
            }
            
            response = await self.llm.analyze_context(prompt)
            steps = response.get('steps', [])
            print(f"Generated steps: {json.dumps(steps, indent=2)}")
            return steps
            
        except Exception as e:
            print(f"Error generating solution steps: {str(e)}")
            print(f"Error traceback: {traceback.format_exc()}")
            return []
        
    async def _extract_context(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context information from task data"""
        print("Starting _extract_context...")
        
        # Get first training example for shape analysis
        first_example = task_data['train'][0]
        input_grid = np.array(first_example['input'])
        output_grid = np.array(first_example['output'])
        
        try:
            # Prepare context for LLM
            context_prompt = {
                'input_shape': input_grid.shape,
                'output_shape': output_grid.shape,
                'input_values': sorted(list(set(input_grid.flatten().tolist()))),
                'output_values': sorted(list(set(output_grid.flatten().tolist()))),
                'num_examples': len(task_data['train'])
            }
            
            # Get LLM insights about the context
            response = await self.llm.analyze_context(context_prompt)
            
            # Combine basic context with LLM insights
            context = {
                'input_shape': input_grid.shape,
                'output_shape': output_grid.shape,
                'input_values': sorted(list(set(input_grid.flatten().tolist()))),
                'output_values': sorted(list(set(output_grid.flatten().tolist()))),
                'num_examples': len(task_data['train']),
                'insights': response.get('insights', []),
                'patterns': response.get('patterns', []),
                'complexity_estimate': response.get('complexity_estimate', 0.5)
            }
            
            return context
            
        except Exception as e:
            print(f"Error in context extraction: {str(e)}")
            # Return basic context if LLM analysis fails
            return {
                'input_shape': input_grid.shape,
                'output_shape': output_grid.shape,
                'input_values': sorted(list(set(input_grid.flatten().tolist()))),
                'output_values': sorted(list(set(output_grid.flatten().tolist()))),
                'num_examples': len(task_data['train'])
            }
            
    async def _extract_unified_context(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract unified context from multiple examples"""
        print("Starting _extract_unified_context...")
        
        # Format examples for prompt
        examples_str = json.dumps(examples, indent=2)
        
        prompt = f"""Analyze these examples and extract unified context that applies to ALL examples.
Return your analysis in the following JSON format:
{{
    "task_type": "type of task",
    "complexity": 0.0-1.0,
    "identified_patterns": ["list of patterns that work for ALL examples"],
    "required_concepts": ["list of required concepts"],
    "constraints": {{"constraint_name": "value"}}
}}

Examples to analyze:
{examples_str}"""
        
        response = await self.llm.get_completion(prompt)
        try:
            print("Finished _extract_unified_context")
            return json.loads(response)
        except (json.JSONDecodeError, KeyError):
            print("Error parsing LLM response")
            return {}
        
    def _check_feedback_needed(self, attempt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if feedback is needed for this learning attempt"""
        print("Starting _check_feedback_needed...")
        # New concept with low confidence
        for item in attempt.get('learned_items', []):
            if item['type'] == 'concept':
                concept = item['content']
                if concept.get('confidence', 1.0) < 0.7:
                    print("Feedback needed for concept validation")
                    return {
                        'type': 'concept_validation',
                        'query': f"Please validate this newly discovered concept:\n{concept['description']}\n\nIs this a valid and useful concept for solving ARC tasks?"
                    }

        # New primitive needs validation
        for item in attempt.get('learned_items', []):
            if item['type'] == 'primitive':
                primitive = item['content']
                if primitive.get('confidence', 1.0) < 0.7:
                    print("Feedback needed for primitive validation")
                    return {
                        'type': 'primitive_validation',
                        'query': f"Please validate this newly discovered primitive:\n{primitive['description']}\n\nIs this a valid and useful primitive for solving ARC tasks?"
                    }

        print("Finished _check_feedback_needed")
        return None
        
    async def _generate_solution_steps(self, task_data: Dict[str, Any], strategy: Dict[str, Any],
                                    concept_application: Dict[str, Any], primitive_suggestions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate solution steps based on strategy, concepts, and primitives"""
        steps = []
        
        # Use strategy steps if available
        if strategy and 'steps' in strategy:
            steps.extend(strategy['steps'])
            
        # Add steps from concept application
        if concept_application and 'steps' in concept_application:
            steps.extend(concept_application['steps'])
            
        # Add steps from primitive suggestions
        if primitive_suggestions and 'suggestions' in primitive_suggestions:
            for suggestion in primitive_suggestions['suggestions']:
                steps.append({
                    'type': suggestion['primitive_id'],
                    'description': f"Apply {suggestion['primitive_id']} primitive",
                    'parameters': suggestion.get('parameters', {})
                })
                
        return steps

    def incorporate_feedback(self, feedback: Dict[str, Any]) -> None:
        """Incorporate feedback into learning system"""
        print("Starting incorporate_feedback...")
        try:
            # Track feedback
            self.feedback_history.append({
                'timestamp': datetime.now().isoformat(),
                'feedback': feedback
            })
            
            # Update strategy performance
            if 'strategy_id' in feedback and 'success_rate' in feedback:
                self.meta_strategy.update_performance(
                    feedback['strategy_id'],
                    feedback.get('context', {}),
                    feedback['success_rate'],
                    feedback
                )
                
            # Track what we learned
            self.learning_history.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'feedback_incorporated',
                'details': feedback
            })

        except Exception as e:
            print(f"Error incorporating feedback: {str(e)}")

    def _generate_meta_insights(self, task_data: Dict[str, Any], result: Dict[str, Any]):
        """Generate meta-insights from task results"""
        success = result.get('success', False)
        accuracy = result.get('accuracy', 0.0)
        
        # Analyze performance patterns
        if success and accuracy > 0.95:
            self.learning_state.record_meta_insight(
                "High accuracy achieved through combination of " +
                f"primitive operations: {', '.join(result.get('primitives_used', []))}",
                importance=0.85
            )
            
        # Analyze failure patterns
        if not success and result.get('failure_reason'):
            self.learning_state.record_meta_insight(
                f"Task failed due to {result['failure_reason']}. " +
                "Consider enhancing primitive operations in this area.",
                importance=0.8
            )
            
        # Analyze strategy effectiveness
        if result.get('strategy_adaptations'):
            self.learning_state.record_meta_insight(
                "Strategy adaptations were necessary for success: " +
                f"{result['strategy_adaptations']}",
                importance=0.75
            )
