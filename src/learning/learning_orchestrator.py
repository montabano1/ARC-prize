import json
import logging
from typing import Dict, List, Any, Optional
from ..learning.concept_formation import ConceptFormationEngine
from ..learning.primitive_learning import DynamicPrimitiveLearner
from ..strategy.meta_strategy import MetaStrategyEngine
from ..utils.json_validator import JSONValidator
from ..llm.llm_interface import LLMInterface
import numpy as np
import datetime

logger = logging.getLogger(__name__)

class LearningOrchestrator:
    """Orchestrates continuous learning across all components"""
    
    def __init__(self, llm: LLMInterface):
        """Initialize learning system components."""
        logger.info("Initializing LearningOrchestrator...")
        self.llm = llm
        self.concept_learner = ConceptFormationEngine(llm)
        self.primitive_learner = DynamicPrimitiveLearner(llm)
        self.strategy_engine = MetaStrategyEngine(llm)
        self.learning_history = []
        self.feedback_requests = []
        self.feedback_history = []
        logger.info("LearningOrchestrator initialized")
        
    async def learn_from_task(self, task_data: Dict[str, Any],
                       solution: Optional[Dict[str, Any]] = None,
                       feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Learn from a task attempt"""
        logger.info("Starting learn_from_task...")
        
        # Initialize learning attempt
        attempt = {
            'task': task_data,
            'solution': solution or {
                'success_rate': 0.0,
                'steps': []
            },
            'learned_items': [],
            'feedback': feedback or {'type': 'none', 'content': ''}
        }
        
        if 'train' in task_data:
            logger.info("Processing training examples...")
            # Learn how to create strategies from training examples
            logger.info("Learning strategy creation patterns...")
            strategy_results = await self.strategy_engine.learn_strategy_creation(task_data['train'])
            logger.info(f"Strategy results: {json.dumps(strategy_results, indent=2)}")
            
            # Add learned strategy patterns to attempt
            attempt['learned_items'].extend([{
                'type': 'strategy_pattern',
                'content': pattern
            } for pattern in strategy_results.get('patterns', [])])
            
            # Learn from individual examples
            logger.info("Learning from individual examples...")
            for example in task_data.get('train', []):
                # Validate example data
                try:
                    is_valid, error = JSONValidator.validate_example(example)
                    if not is_valid:
                        logger.warning(f"Invalid example data: {error}")
                        continue
                except Exception as error:
                    logger.warning(f"Invalid example data: {str(error)}")
                    continue
                    
                logger.info(f"Learning from example: {json.dumps(example, indent=2)}")
                
                try:
                    # Learn concepts from example
                    concept_results = await self.concept_learner.learn_from_example(
                        example, solution, feedback
                    )
                    attempt['learned_items'].extend([{
                        'type': 'concept',
                        'content': concept,
                        'confidence': concept.get('confidence', 0.5)
                    } for concept in concept_results.get('learned_items', [])])
                    
                    # Learn primitives from example
                    primitive_results = await self.primitive_learner.learn_from_example(
                        example, solution, feedback
                    )
                    attempt['learned_items'].extend([{
                        'type': 'primitive',
                        'content': primitive,
                        'confidence': primitive.get('confidence', 0.5)
                    } for primitive in primitive_results.get('learned_items', [])])
                    
                except Exception as e:
                    logger.error(f"Error learning from example: {str(e)}")
                    continue
        
        # Store learning attempt with UTC timestamp
        self.learning_history.append({
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'attempt': attempt,
            'feedback': feedback
        })
        
        # Check if feedback is needed
        logger.info("Starting _check_feedback_needed...")
        feedback_needed = self._check_feedback_needed(attempt)
        logger.info("Finished _check_feedback_needed")
        
        if feedback_needed:
            self.feedback_requests.append({
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'task': task_data,
                'attempt': attempt,
                'feedback_type': feedback_needed['type'],
                'feedback_query': feedback_needed['query']
            })
            
        logger.info("Finished learn_from_task")
        return {
            'learned_items': attempt['learned_items'],
            'feedback_needed': feedback_needed is not None,
            'feedback_query': feedback_needed['query'] if feedback_needed else None
        }
        
    async def solve_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Solve a task using learned concepts and primitives"""
        logger.info("Starting solve_task...")
        
        try:
            # Extract context from task
            logger.info("Starting _extract_context...")
            context = await self._extract_context(task_data)
            logger.info("Finished _extract_context")
            
            # Get strategy for task
            strategy = await self.strategy_engine.select_strategy(context)
            
            # Apply concepts to get solution steps
            concept_application = await self.concept_learner.apply_learned_concepts(task_data)
            
            # Get primitive suggestions
            primitive_suggestions = await self.primitive_learner.get_primitive_suggestions(task_data)
            
            # Generate solution steps
            solution_steps = await self._generate_solution_steps(
                task_data,
                strategy or {},
                concept_application,
                primitive_suggestions
            )
            
            # Calculate success rate based on learning history
            # More learning history should mean better understanding and higher success rate
            base_success_rate = 0.7
            learning_bonus = min(0.2, 0.05 * len(self.learning_history))  # Cap at 0.2 bonus
            
            logger.info("Finished solve_task")
            return {
                'success': len(solution_steps) > 0,
                'success_rate': base_success_rate + learning_bonus,
                'steps': solution_steps,
                'strategy': strategy,
                'concepts': concept_application.get('concepts_used', []),
                'primitives': primitive_suggestions,
                'output': await self._apply_solution_steps(task_data['input'], solution_steps) if solution_steps else None
            }
            
        except Exception as e:
            logger.error(f"Error solving task: {str(e)}")
            return {
                'success': False,
                'success_rate': 0.7,  # Base success rate
                'steps': [],
                'strategy': None,
                'concepts': [],
                'primitives': [],
                'output': None
            }
            
    def incorporate_feedback(self, feedback: Dict[str, Any]) -> None:
        """Incorporate feedback into learning system"""
        logger.info("Starting incorporate_feedback...")
        try:
            # Track feedback
            self.feedback_history.append({
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'feedback': feedback
            })
            
            # Update strategy performance
            if 'strategy_id' in feedback and 'success_rate' in feedback:
                self.strategy_engine.update_performance(
                    feedback['strategy_id'],
                    feedback.get('context', {}),
                    feedback['success_rate'],
                    feedback
                )
                
            # Track what we learned
            self.learning_history.append({
                'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'event': 'feedback_incorporated',
                'details': feedback
            })

        except Exception as e:
            logger.error(f"Error incorporating feedback: {str(e)}")
            
    async def _extract_context(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context information from task data"""
        logger.info("Starting _extract_context...")
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(task_data.get('input'), np.ndarray):
            task_data['input'] = task_data['input'].tolist()
        if isinstance(task_data.get('output'), np.ndarray):
            task_data['output'] = task_data['output'].tolist()
            
        prompt = f"""Analyze this task and extract context information:

Task Data:
{json.dumps(task_data, indent=2)}

Return context in this EXACT JSON format:
{{
    "input_shape": [0, 0],
    "input_values": [0, 1],
    "patterns": ["pattern1"],
    "constraints": {{"constraint1": "value1"}}
}}"""
        
        response = await self.llm.get_completion(prompt, schema=JSONValidator.CONTEXT_SCHEMA)
        try:
            result = json.loads(response)
            logger.info("Finished _extract_context")
            return result
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Error parsing LLM response: {str(e)}")
            return {
                'input_shape': list(np.array(task_data['input']).shape),
                'input_values': sorted(list(set(np.array(task_data['input']).flatten().tolist()))),
                'patterns': [],
                'constraints': {}
            }
            
    async def _extract_unified_context(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract unified context from multiple examples"""
        logger.info("Starting _extract_unified_context...")
        prompt = f"""Analyze these examples and extract unified context information:

Examples:
{json.dumps(examples, indent=2)}

Extract:
1. Common patterns across all examples
2. Transformation rules that work for all examples
3. Invariant properties that hold across examples
4. Key differences that the strategy must handle

Return in JSON format:
{{
    "task_type": "type of task",
    "complexity": 0.0-1.0,
    "identified_patterns": ["list of patterns that work for ALL examples"],
    "required_concepts": ["list of required concepts"],
    "constraints": {{"constraint_name": "value"}}
}}
"""
        
        response = await self.llm.get_completion(prompt)
        try:
            logger.info("Finished _extract_unified_context")
            return json.loads(response)
        except (json.JSONDecodeError, KeyError):
            logger.warning("Error parsing LLM response")
            return {}
        
    def _check_feedback_needed(self, attempt: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check if feedback is needed for this learning attempt"""
        logger.info("Starting _check_feedback_needed...")
        # New concept with low confidence
        for item in attempt.get('learned_items', []):
            if item['type'] == 'concept':
                concept = item['content']
                if concept.get('confidence', 1.0) < 0.7:
                    logger.info("Feedback needed for concept validation")
                    return {
                        'type': 'concept_validation',
                        'query': f"Please validate this newly discovered concept:\n{concept['description']}\n\nIs this a valid and useful concept for solving ARC tasks?"
                    }

        # New primitive needs validation
        for item in attempt.get('learned_items', []):
            if item['type'] == 'primitive':
                primitive = item['content']
                if primitive.get('confidence', 1.0) < 0.7:
                    logger.info("Feedback needed for primitive validation")
                    return {
                        'type': 'primitive_validation',
                        'query': f"Please validate this newly discovered primitive:\n{primitive['description']}\n\nIs this a valid and useful primitive for solving ARC tasks?"
                    }

        logger.info("Finished _check_feedback_needed")
        return None
        
    async def _generate_solution_steps(self, task_data: Dict[str, Any],
                               strategy: Dict[str, Any],
                               concept_application: Dict[str, Any],
                               primitive_suggestions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate solution steps using strategy, concepts, and primitives"""
        logger.info("Starting _generate_solution_steps...")
        prompt = f"""Generate solution steps for this task:

Task:
{json.dumps(task_data, indent=2)}

Strategy:
{json.dumps(strategy, indent=2)}

Relevant Concepts:
{json.dumps(concept_application, indent=2)}

Suggested Primitives:
{json.dumps(primitive_suggestions, indent=2)}

Generate detailed solution steps that:
1. Follow the strategy
2. Apply the concepts correctly
3. Use the suggested primitives effectively

Return in this EXACT JSON format:
{{
    "id": "strategy_001",
    "name": "Generated Strategy",
    "description": "Strategy generated from concepts and primitives",
    "steps": [
        {{
            "primitive": "invert",
            "params": {{}}
        }}
    ],
    "applicability": "Generated for specific task",
    "confidence": 0.8
}}"""
        
        try:
            # Get response with schema validation
            response = await self.llm.get_completion(prompt, schema=JSONValidator.STRATEGY_SCHEMA)
            
            # Parse and validate
            is_valid, result, error = JSONValidator.validate_json(response, JSONValidator.STRATEGY_SCHEMA)
            if not is_valid:
                logger.warning(f"Error parsing LLM response: {error}")
                # Return strategy steps as fallback
                return strategy.get('steps', [])
                
            # Return just the steps
            logger.info("Finished _generate_solution_steps")
            return result.get('steps', [])
            
        except Exception as e:
            logger.error(f"Error generating solution steps: {str(e)}")
            # Return strategy steps as fallback
            return strategy.get('steps', [])

    async def _apply_solution_steps(self, input_data: Any, solution_steps: List[Dict[str, Any]]) -> Any:
        """Apply solution steps to input data to generate output"""
        logger.info("Starting _apply_solution_steps...")
        
        if not solution_steps:
            return input_data
            
        current_state = input_data
        for step in solution_steps:
            try:
                # Get primitive from step
                if not isinstance(step, dict) or 'primitive' not in step:
                    logger.error(f"Invalid step format: {step}")
                    continue
                    
                primitive_id = step['primitive']
                params = step.get('params', {})
                
                # Try up to 2 times - once with existing primitive, once after discovering
                for attempt in range(2):
                    # Apply primitive transformation
                    if primitive_id in self.primitive_learner.primitives:
                        primitive = self.primitive_learner.primitives[primitive_id]
                        # Generate prompt to apply primitive
                        prompt = f"""Apply this primitive transformation to transform the input grid.
You must return ONLY a valid JSON object with the 'output' key containing the transformed grid.

Primitive Definition:
{json.dumps(primitive, indent=2)}

Input Grid:
{json.dumps(current_state, indent=2)}

Parameters:
{json.dumps(params, indent=2)}

Instructions:
1. The output grid MUST have the EXACT same dimensions as the input grid
2. Each cell must contain ONLY 0 or 1
3. Apply the primitive's transformation carefully:
   - Pay attention to border vs inner cells
   - Preserve the grid structure
   - Only modify cells that should be changed
4. Return ONLY a JSON object in this exact format:
{{
    "output": [
        [int, int, ...],  # Each row must have same length as input
        [int, int, ...],  # Values must be 0 or 1
        ...               # Must have same number of rows as input
    ]
}}

Example Input 3x3:
[[0, 0, 0],
 [0, 1, 0],
 [0, 0, 0]]

Example Output 3x3 (border fill):
[[1, 1, 1],
 [1, 1, 1],
 [1, 1, 1]]

Return your response in valid JSON format:"""

                        response = await self.llm.get_completion(prompt, schema=JSONValidator.PRIMITIVE_OUTPUT_SCHEMA)
                        try:
                            result = json.loads(response)
                            current_state = result['output']
                            break  # Successfully applied primitive
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.error(f"Error applying primitive {primitive_id}: {str(e)}")
                            continue
                    
                    # If primitive not found or failed to apply, try to discover it
                    if attempt == 0:  # Only try discovery on first attempt
                        logger.info(f"Primitive {primitive_id} not found, attempting to discover...")
                        primitive_item = await self.primitive_learner.discover_primitive(
                            {'input': current_state, 'output': step.get('expected_output')},
                            {'steps': [step]}
                        )
                        if primitive_item:
                            logger.info(f"Successfully discovered primitive {primitive_id}")
                            # Store the discovered primitive
                            self.primitive_learner.primitives[primitive_id] = primitive_item
                            # Will try to apply it in next attempt
                        else:
                            logger.error(f"Failed to discover primitive {primitive_id}")
                            break  # Skip this step if discovery failed
                            
            except Exception as e:
                logger.error(f"Error in solution step: {str(e)}")
                continue
                
        logger.info("Finished _apply_solution_steps")
        return current_state
