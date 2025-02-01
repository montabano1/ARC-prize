from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
import json
from src.dsl.primitives import DynamicPrimitiveLibrary  # Import DynamicPrimitiveLibrary
from src.llm.llm_interface import LLMInterface
from src.utils.json_validator import JSONValidator
import time
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PrimitiveEvolution:
    """Tracks how a primitive evolves over time"""
    primitive_id: str
    versions: List[Dict[str, Any]]
    performance_history: List[float]
    usage_contexts: List[Dict[str, Any]]
    adaptations: List[Dict[str, Any]]

class DynamicPrimitiveLearner:
    """System for learning new primitive operations"""
    
    def __init__(self, llm: LLMInterface, state_manager=None):
        self.llm = llm
        self.state_manager = state_manager
        self.primitive_library = DynamicPrimitiveLibrary(llm)  # Initialize primitive library
        self.primitives = self.primitive_library.primitives  # Use primitives from library
        self.evolution_history = {}  # id -> list of changes
        self.primitive_combinations = []  # Successful combinations
        self.failed_attempts = []  # Failed primitive attempts
        self.feedback_threshold = 0.7
        self.primitive_stats = {}
        
    async def discover_primitive(self, task_data: Dict[str, Any], solution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Discover new primitive operations from successful solutions"""
        # Analyze solution to identify potential new primitives
        # Generate primitive prompt
        prompt = f"""Analyze this task and solution to discover a new primitive operation.
The primitive should be a focused, single-purpose transformation that can be applied to a grid.

Input Grid:
{json.dumps(task_data['input'], indent=2)}

Expected Output:
{json.dumps(task_data['output'], indent=2)}

Solution Steps:
{json.dumps(solution, indent=2)}

Instructions:
1. Identify a single, focused transformation that helps achieve the output
2. The primitive should be simple and reusable
3. Include clear parameters that control the transformation
4. Provide specific examples showing input->output
5. Return the primitive in this exact JSON format:
{{
    "primitive": {{
        "id": "primitive_id",              # Unique identifier
        "name": "Descriptive Name",        # Clear, descriptive name
        "description": "What it does",     # Detailed explanation
        "parameters": {{                   # Parameters that control the transformation
            "param_name": "description"
        }},
        "implementation_guide": "Step by step implementation instructions",
        "applicability": "When to use this primitive",
        "examples": [
            "input: [[0,0],[0,0]] -> output: [[1,1],[1,1]] # Fill all with 1",
            "input: [[1,1],[1,1]] -> output: [[0,0],[0,0]] # Fill all with 0"
        ]
    }}
}}

Example primitive for border fill:
{{
    "primitive": {{
        "id": "fill_border",
        "name": "Fill Border Values",
        "description": "Sets all border cells (outer edges) of the grid to a specified value",
        "parameters": {{
            "border_value": "Value to set border cells to (0 or 1)"
        }},
        "implementation_guide": "1. Get grid dimensions\\n2. Set first and last row to border_value\\n3. Set first and last column to border_value\\n4. Leave inner values unchanged",
        "applicability": "When you need to create a border or frame around the grid",
        "examples": [
            "input: [[0,0,0],[0,1,0],[0,0,0]] -> output: [[1,1,1],[1,1,1],[1,1,1]] # Fill border with 1s",
            "input: [[1,1,1],[1,0,1],[1,1,1]] -> output: [[0,0,0],[0,0,0],[0,0,0]] # Fill border with 0s"
        ]
    }}
}}

Return your primitive definition in valid JSON format:"""

        max_retries = 3
        current_retry = 0
        
        while current_retry < max_retries:
            try:
                # Get LLM response with schema validation
                response = await self.llm.get_completion(prompt, schema=JSONValidator.PRIMITIVE_SCHEMA)
                is_valid, result, error = JSONValidator.validate_json(response, JSONValidator.PRIMITIVE_SCHEMA)
                
                if not is_valid:
                    # Generate fix prompt and retry
                    fix_prompt = JSONValidator.generate_fix_prompt(prompt, error, JSONValidator.PRIMITIVE_SCHEMA)
                    response = await self.llm.get_completion(fix_prompt, schema=JSONValidator.PRIMITIVE_SCHEMA)
                    is_valid, result, error = JSONValidator.validate_json(response, JSONValidator.PRIMITIVE_SCHEMA)
                    
                    if not is_valid:
                        current_retry += 1
                        continue
                
                primitive = result['primitive']

                # Validate primitive matches schema
                primitive_item = {
                    'id': primitive['id'],
                    'name': primitive['name'],
                    'description': primitive['description'],
                    'parameters': primitive.get('parameters', {}),
                    'implementation_guide': primitive.get('implementation_guide', ''),
                    'applicability': primitive.get('applicability', ''),
                    'examples': primitive.get('examples', [])
                }

                # Validate learning result
                learning_result = {
                    'learned_items': [{
                        'type': 'primitive',
                        'item': primitive_item
                    }]
                }

                is_valid, _, error = JSONValidator.validate_json(json.dumps(learning_result), JSONValidator.LEARNING_RESULT_SCHEMA)
                if not is_valid:
                    print(f"Invalid primitive learning result: {error}")
                    current_retry += 1
                    continue

                # Record primitive learning in state manager
                if self.state_manager:
                    primitive_data = {
                        'id': f"primitive_{primitive['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'name': primitive['name'],
                        'success': True,
                        'context': task_data,
                        'stats': primitive_item
                    }
                    self.state_manager.record_pattern(primitive_data, success=True)
                    
                return primitive_item

            except Exception as e:
                print(f"Error discovering primitive: {str(e)}")
                current_retry += 1
                
        return None

    async def evaluate_primitive(self, primitive_id: str,
                         performance: float,
                         context: Dict[str, Any]) -> None:
        """Evaluate primitive performance and trigger adaptation if needed"""
        if primitive_id not in self.primitives:
            return
            
        evolution = self.evolution_history[primitive_id]
        
        # Record performance and context
        evolution.append({
            'timestamp': time.time(),
            'event': 'evaluation',
            'details': {
                'performance': performance,
                'context': context
            }
        })
        
        # Check if adaptation is needed
        if len(evolution) >= 5:
            recent_performance = [e['details']['performance'] for e in evolution[-5:]]
            if np.mean(recent_performance) < 0.7:
                await self._adapt_primitive(primitive_id)
                
    async def combine_primitives(self, primitives: List[str],
                         task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try to create new primitive by combining existing ones"""
        primitive_details = []
        for p_id in primitives:
            if p_id in self.primitives:
                primitive_details.append(self.primitives[p_id])
                
        if not primitive_details:
            return None
            
        prompt = f"""Analyze these primitives and suggest how to combine them:

Primitives:
{json.dumps(primitive_details, indent=2)}

Task Context:
{json.dumps(task, indent=2)}

Suggest a new primitive that combines these effectively.
Return in JSON format:
{{
    "combined_primitive": {{
        "id": "unique_id",
        "name": "primitive name",
        "description": "what it does",
        "parameters": {{"param_name": "param_type"}},
        "implementation_guide": "how to implement it",
        "component_primitives": ["list of primitive ids used"],
        "composition_logic": "how primitives are combined"
    }}
}}"""
        
        response = await self.llm.get_completion(prompt, schema=JSONValidator.COMBINED_PRIMITIVE_SCHEMA)
        try:
            result = json.loads(response)
            new_primitive = result['combined_primitive']
            
            # Record successful combination
            self.primitive_combinations.append({
                'components': primitives,
                'result': new_primitive,
                'task_context': task
            })
            
            # Track new primitive
            if new_primitive['id'] not in self.primitives:
                self.primitives[new_primitive['id']] = new_primitive
                self.evolution_history[new_primitive['id']] = [{
                    'timestamp': time.time(),
                    'event': 'created',
                    'details': new_primitive
                }]
                
            return new_primitive
        except (json.JSONDecodeError, KeyError):
            return None
            
    async def get_primitive_suggestions(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get primitive suggestions for a task"""
        # Analyze task and find relevant primitives
        available_primitives = list(self.primitives.values())
        
        prompt = f"""Suggest primitives for this task:

Task:
{json.dumps(task, indent=2)}

Available Primitives:
{json.dumps(available_primitives, indent=2)}

Consider:
1. Direct matches based on task requirements
2. Combinations that might work together
3. Primitives that might need adaptation

Return in JSON format:
{{
    "suggestions": [
        {{
            "primitive_id": "id",
            "confidence": 0.0-1.0,
            "rationale": "why this primitive is suitable"
        }}
    ]
}}"""
        
        response = await self.llm.get_completion(prompt, schema=JSONValidator.PRIMITIVE_SUGGESTION_SCHEMA)
        try:
            result = json.loads(response)
            
            # Filter suggestions by confidence
            suggestions = [
                sugg for sugg in result['suggestions']
                if sugg['confidence'] > 0.7
            ]
            
            # Try to find combinations for high-confidence primitives
            high_conf_primitives = [sugg['primitive_id'] for sugg in suggestions if sugg['confidence'] > 0.8]
            if len(high_conf_primitives) >= 2:
                new_primitive = await self.combine_primitives(high_conf_primitives[:2], task)
                if new_primitive:
                    suggestions.append({
                        'primitive_id': new_primitive['id'],
                        'confidence': 0.8,  # Default score for combinations
                        'rationale': f"Combined primitive created from {', '.join(high_conf_primitives[:2])}"
                    })
                    
            return suggestions
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error getting primitive suggestions: {str(e)}")
            return []

    async def get_applicable_primitives(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get primitives that are applicable to this task"""
        applicable_primitives = []
        
        # Generate prompt to analyze task and find applicable primitives
        prompt = f"""Analyze this task and determine which primitives would be applicable:

Task:
{json.dumps(task_data, indent=2)}

Available primitives:
{json.dumps(list(self.primitives.values()), indent=2)}

Return list of applicable primitives with explanations:
{{
    "applicable_primitives": [
        {{
            "primitive_id": "id of applicable primitive",
            "applicability_score": 0.0-1.0,
            "explanation": "why this primitive is applicable"
        }}
    ]
}}"""

        try:
            response = await self.llm.get_completion(prompt)
            result = json.loads(response)
            
            # Get full primitive details for each applicable primitive
            for item in result.get('applicable_primitives', []):
                primitive_id = item['primitive_id']
                if primitive_id in self.primitives:
                    primitive = self.primitives[primitive_id].copy()
                    primitive['applicability_score'] = item['applicability_score']
                    primitive['applicability_explanation'] = item['explanation']
                    applicable_primitives.append(primitive)
                    
            return sorted(applicable_primitives, 
                        key=lambda x: x.get('applicability_score', 0),
                        reverse=True)
                        
        except Exception as e:
            print(f"Error finding applicable primitives: {str(e)}")
            return []

    async def learn_from_example(self, task_data: Dict[str, Any],
                         solution: Optional[Dict[str, Any]] = None,
                         feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Learn primitive operations from an example"""
        learned_items = []
        
        # First try to discover new primitives
        if solution:
            primitive = await self.discover_primitive(task_data, solution)
            if primitive:
                learned_items.append({
                    'type': 'primitive',
                    'content': primitive,
                    'confidence': primitive.get('confidence', 0.5)
                })
                
        # Then try to combine existing primitives
        if len(self.primitives) >= 2:
            primitive_ids = list(self.primitives.keys())[:2]  # Start with just 2 primitives
            combined = await self.combine_primitives(primitive_ids, task_data)
            if combined:
                learned_items.append({
                    'type': 'primitive_combination',
                    'content': combined,
                    'confidence': combined.get('confidence', 0.5)
                })
                
        # Get suggestions for primitive usage
        suggestions = await self.get_primitive_suggestions(task_data)
        if suggestions:
            learned_items.append({
                'type': 'primitive_suggestions',
                'content': suggestions,
                'confidence': 0.8  # High confidence since these are based on existing primitives
            })
            
        return {
            'learned_items': learned_items,
            'feedback_needed': len(learned_items) > 0 and any(item['confidence'] < self.feedback_threshold for item in learned_items)
        }

    async def _track_evolution(self, primitive: Dict[str, Any],
                        solution: Optional[Dict[str, Any]] = None,
                        feedback: Optional[Dict[str, Any]] = None) -> None:
        """Track how a primitive evolves"""
        if primitive['id'] not in self.evolution_history:
            self.evolution_history[primitive['id']] = PrimitiveEvolution(
                primitive_id=primitive['id'],
                versions=[primitive],
                performance_history=[],
                usage_contexts=[],
                adaptations=[]
            )
            
        evolution = self.evolution_history[primitive['id']]
        
        # Add solution performance
        if solution:
            evolution.performance_history.append(solution.get('success_rate', 0.0))
            evolution.usage_contexts.append({
                'task': solution.get('task', {}),
                'steps': solution.get('steps', []),
                'timestamp': time.time()
            })
            
        # Add feedback
        if feedback:
            evolution.adaptations.append({
                'feedback': feedback,
                'timestamp': time.time()
            })
            
        # Check if adaptation needed
        if len(evolution.performance_history) >= 5:
            recent_performance = evolution.performance_history[-5:]
            if np.mean(recent_performance) < 0.7:
                adaptation = await self._generate_adaptation(primitive, {
                    'trigger': 'poor_performance',
                    'performance': recent_performance
                })
                if adaptation:
                    evolution.versions.append(adaptation)
                    evolution.adaptations.append({
                        'trigger': 'poor_performance',
                        'adaptation': adaptation
                    })
                    
    async def _generate_adaptation(self, primitive: Dict[str, Any],
                           trigger: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate adaptation for a primitive"""
        prompt = f"""This primitive needs adaptation:

Primitive:
{json.dumps(primitive, indent=2)}

Trigger:
{json.dumps(trigger, indent=2)}

Return adapted primitive in this format:
{{
    "primitive": {{
        "id": "{primitive['id']}_v{len(self.evolution_history[primitive['id']].versions) + 1}",
        "name": "adapted primitive name",
        "description": "what this adapted primitive does",
        "parameters": {{"param_name": "param description"}},
        "implementation_guide": "how to implement this adapted primitive",
        "applicability": "when to use this adapted primitive",
        "adaptations": ["what was changed"]
    }}
}}"""

        try:
            response = await self.llm.get_completion(prompt, schema=JSONValidator.PRIMITIVE_SCHEMA)
            result = json.loads(response)
            return result['primitive']
        except Exception as e:
            print(f"Error generating adaptation: {str(e)}")
            return None

    async def _check_feedback_needed(self, primitives: List[Dict[str, Any]]) -> bool:
        """Check if human feedback is needed"""
        for primitive in primitives:
            evolution = self.evolution_history.get(primitive['id'])
            if evolution:
                # Check recent performance
                if len(evolution.performance_history) >= 3:
                    recent_performance = evolution.performance_history[-3:]
                    if np.mean(recent_performance) < self.feedback_threshold:
                        return True
                        
                # Check adaptation history
                if len(evolution.adaptations) >= 2:
                    recent_adaptations = evolution.adaptations[-2:]
                    if all('poor_performance' in adapt['trigger'] for adapt in recent_adaptations):
                        return True
                        
        return False

    async def _apply_feedback(self, primitives: List[Dict[str, Any]], 
                       feedback: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply feedback to primitives"""
        updated_primitives = []
        for primitive in primitives:
            try:
                # Generate improvement prompt
                prompt = f"""Apply this feedback to improve the primitive:

Primitive:
{json.dumps(primitive, indent=2)}

Feedback:
{json.dumps(feedback, indent=2)}

Return improved primitive in the same format as the input."""

                response = await self.llm.get_completion(prompt, schema=JSONValidator.PRIMITIVE_SCHEMA)
                result = json.loads(response)
                
                if result.get('primitive'):
                    updated_primitives.append(result['primitive'])
                    
            except Exception as e:
                print(f"Error applying feedback: {str(e)}")
                continue
                
        return updated_primitives

    async def _adapt_primitive(self, primitive_id: str) -> None:
        """Adapt a primitive based on performance history"""
        if primitive_id not in self.primitives:
            return
            
        evolution = self.evolution_history[primitive_id]
        current_version = self.primitives[primitive_id]
        
        # Analyze usage patterns and performance
        prompt = f"""This primitive needs adaptation:

Current Version:
{json.dumps(current_version, indent=2)}

Performance History:
{json.dumps({
    'performance': [e['details']['performance'] for e in evolution[-5:]],
    'contexts': [e['details']['context'] for e in evolution[-5:]]
}, indent=2)}

Suggest adaptations in JSON format:
{{
    "adapted_primitive": {{
        "id": "{primitive_id}_v{len(evolution)}",
        "name": "primitive name",
        "description": "what it does",
        "parameters": {{"param_name": "param_type"}},
        "implementation_guide": "how to implement it",
        "applicability": "when to use it",
        "adaptations": ["what was changed"]
    }}
}}"""
        
        response = await self.llm.get_completion(prompt, schema=JSONValidator.ADAPTED_PRIMITIVE_SCHEMA)
        try:
            result = json.loads(response)
            adapted = result['adapted_primitive']
            
            # Record adaptation
            evolution.append({
                'timestamp': time.time(),
                'event': 'adaptation',
                'details': adapted
            })
            self.primitives[primitive_id] = adapted
            
        except (json.JSONDecodeError, KeyError):
            pass

    def learn_from_success(self, task_data: Dict[str, Any], result: Dict[str, Any]):
        """Learn from a successful task execution"""
        primitives_used = result.get('primitives_used', [])
        context = {
            'input_size': task_data.get('input_size'),
            'output_size': task_data.get('output_size'),
            'task_type': task_data.get('task_type')
        }
        
        for primitive in primitives_used:
            if primitive not in self.primitives:
                self.primitives[primitive] = {
                    'name': primitive,
                    'success_count': 0,
                    'total_uses': 0,
                    'contexts': []
                }
                
            # Update primitive stats
            self.primitives[primitive]['success_count'] += 1
            self.primitives[primitive]['total_uses'] += 1
            self.primitives[primitive]['contexts'].append(context)
            
            # Record primitive learning in state manager
            if self.state_manager:
                primitive_data = {
                    'id': f"primitive_{primitive}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    'name': primitive,
                    'success': True,
                    'context': context,
                    'stats': self.primitives[primitive]
                }
                self.state_manager.record_pattern(primitive_data, success=True)
                
                # Record meta-insight if primitive is highly successful
                success_rate = (self.primitives[primitive]['success_count'] / 
                              self.primitives[primitive]['total_uses'])
                if success_rate > 0.9 and self.primitives[primitive]['total_uses'] > 5:
                    self.state_manager.record_meta_insight(
                        f"Primitive '{primitive}' shows consistent high performance " +
                        f"({success_rate:.2%} success rate over {self.primitives[primitive]['total_uses']} uses) " +
                        f"for {context['task_type']} tasks",
                        importance=0.85
                    )
    
    def get_relevant_primitives(self, task_data: Dict[str, Any]) -> List[str]:
        """Get primitives relevant to the given task"""
        context = {
            'input_size': task_data.get('input_size'),
            'output_size': task_data.get('output_size'),
            'task_type': task_data.get('task_type')
        }
        
        relevant_primitives = []
        for name, stats in self.primitives.items():
            # Check if primitive has been successful in similar contexts
            for past_context in stats['contexts']:
                if self._context_similarity(context, past_context) > 0.7:
                    success_rate = stats['success_count'] / max(1, stats['total_uses'])
                    if success_rate > 0.6:  # Consider primitives with >60% success rate
                        relevant_primitives.append(name)
                        break
                        
        return relevant_primitives
        
    def _context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        similarity = 0.0
        total_weights = 0.0
        
        # Compare input sizes (weight: 0.3)
        if context1.get('input_size') == context2.get('input_size'):
            similarity += 0.3
        total_weights += 0.3
        
        # Compare output sizes (weight: 0.3)
        if context1.get('output_size') == context2.get('output_size'):
            similarity += 0.3
        total_weights += 0.3
        
        # Compare task types (weight: 0.4)
        if context1.get('task_type') == context2.get('task_type'):
            similarity += 0.4
        total_weights += 0.4
        
        return similarity / total_weights if total_weights > 0 else 0.0
