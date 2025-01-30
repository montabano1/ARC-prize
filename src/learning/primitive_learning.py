from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import dataclass
import json
from src.dsl.primitives import DSLPrimitive
from src.llm.llm_interface import LLMInterface
from src.utils.json_validator import JSONValidator
import time

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
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.primitives = {}  # id -> primitive
        self.evolution_history = {}  # id -> list of changes
        self.primitive_combinations = []  # Successful combinations
        self.failed_attempts = []  # Failed primitive attempts
        self.feedback_threshold = 0.7
        
    def discover_primitive(self, task_data: Dict[str, Any], solution: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Discover new primitive operations from successful solutions"""
        # Analyze solution to identify potential new primitives
        # Generate primitive prompt
        prompt = f"""Analyze this task and solution to discover a new primitive operation:

Task input: {json.dumps(task_data['input'], indent=2)}
Task output: {json.dumps(task_data['output'], indent=2)}
Solution steps: {json.dumps(solution, indent=2)}

Return primitive in JSON format:
{{
    "primitive": {{
        "id": "primitive_id",
        "name": "primitive name",
        "description": "what this primitive does",
        "parameters": {{"param_name": "param description"}},
        "implementation_guide": "how to implement this primitive",
        "applicability": "when to use this primitive",
        "examples": [
            "example usage 1",
            "example usage 2"
        ]
    }}
}}"""

        max_retries = 3
        current_retry = 0
        
        while current_retry < max_retries:
            try:
                # Get LLM response with schema validation
                response = self.llm.get_completion(prompt, schema=JSONValidator.PRIMITIVE_SCHEMA)
                is_valid, result, error = JSONValidator.validate_json(response, JSONValidator.PRIMITIVE_SCHEMA)
                
                if not is_valid:
                    # Generate fix prompt and retry
                    fix_prompt = JSONValidator.generate_fix_prompt(prompt, error, JSONValidator.PRIMITIVE_SCHEMA)
                    response = self.llm.get_completion(fix_prompt, schema=JSONValidator.PRIMITIVE_SCHEMA)
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

                # Track primitive evolution
                if primitive['id'] not in self.primitives:
                    self.primitives[primitive['id']] = primitive_item
                    self.evolution_history[primitive['id']] = [{
                        'timestamp': time.time(),
                        'event': 'created',
                        'details': primitive_item
                    }]

                # Return full primitive with all fields
                return primitive_item
                
            except Exception as e:
                print(f"Error discovering primitive (attempt {current_retry + 1}): {str(e)}")
                current_retry += 1
        
        print("Failed to discover primitive after all retries")
        return None
            
    def evaluate_primitive(self, primitive_id: str,
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
                self._adapt_primitive(primitive_id)
                
    def combine_primitives(self, primitives: List[str],
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
        
        response = self.llm.get_completion(prompt, schema=JSONValidator.COMBINED_PRIMITIVE_SCHEMA)
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
            
    def _adapt_primitive(self, primitive_id: str) -> None:
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
        
        response = self.llm.get_completion(prompt, schema=JSONValidator.ADAPTED_PRIMITIVE_SCHEMA)
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
            
    def get_primitive_suggestions(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            "relevance_score": 0.0-1.0,
            "usage_suggestion": "how to use it",
            "potential_adaptations": ["possible adaptations needed"]
        }}
    ],
    "combination_suggestions": [
        {{
            "primitive_ids": ["id1", "id2"],
            "combination_rationale": "why combine these"
        }}
    ]
}}"""
        
        response = self.llm.get_completion(prompt, schema=JSONValidator.PRIMITIVE_SUGGESTIONS_SCHEMA)
        try:
            result = json.loads(response)
            
            # Filter suggestions by relevance
            suggestions = [
                sugg for sugg in result['suggestions']
                if sugg['relevance_score'] > 0.7
            ]
            
            # Add combination suggestions
            for combo in result.get('combination_suggestions', []):
                primitives = combo['primitive_ids']
                new_primitive = self.combine_primitives(primitives, task)
                if new_primitive:
                    suggestions.append({
                        'primitive_id': new_primitive['id'],
                        'relevance_score': 0.8,  # Default score for combinations
                        'usage_suggestion': f"Use this combined primitive: {new_primitive['description']}",
                        'potential_adaptations': []
                    })
                    
            return suggestions
            
        except (json.JSONDecodeError, KeyError):
            return []

    def get_applicable_primitives(self, task_data: Dict[str, Any]) -> List[Dict[str, Any]]:
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
            response = self.llm.get_completion(prompt)
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
