from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass
import json
from src.dsl.primitives import DSLPrimitive
from src.llm.llm_interface import LLMInterface
from src.utils.json_validator import JSONValidator
from src.utils.validators import SystemValidators, ValidationError

@dataclass
class Strategy:
    """Represents a problem-solving strategy"""
    id: str
    name: str
    description: str
    components: List[str]  # List of primitive IDs or sub-strategies
    context: Dict[str, Any]  # When this strategy is applicable
    performance: Dict[str, float]  # Historical performance metrics
    creation_method: str  # How this strategy was created (composed/discovered/evolved)

@dataclass
class Context:
    """Task context information"""
    task_type: str  # e.g., "transformation", "pattern_completion"
    complexity: float
    identified_patterns: List[Dict[str, Any]]
    required_concepts: List[str]
    constraints: Dict[str, Any]

class MetaStrategyEngine:
    """Manages strategy selection, composition, and evolution"""
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.strategies: Dict[str, Strategy] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.context_history: List[Context] = []
        self.adaptation_threshold = 0.7  # Performance threshold for adaptation
        self.used_strategy_ids: Set[str] = set()

    def select_strategy(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select best strategy for context"""
        try:
            # Generate unique ID
            base_id = f"strategy_{len(self.strategies)}"
            strategy_id = base_id
            counter = 1
            while not SystemValidators.validate_strategy_id_unique(strategy_id, self.used_strategy_ids):
                strategy_id = f"{base_id}_{counter}"
                counter += 1

            # Check if we have poor performing strategies to learn from
            poor_strategies = [
                s for s in self.strategies.values()
                if any(p < self.adaptation_threshold for p in s.performance.values())
            ]

            # Generate strategy based on context and past performance
            prompt = f"""Generate a strategy for this context:

Context:
{json.dumps(context, indent=2)}

Past Strategies Performance:
{json.dumps([{
    'strategy': s.name,
    'performance': s.performance,
    'context': s.context
} for s in self.strategies.values()], indent=2)}

Poor Strategies:
{json.dumps([{
    'strategy': s.name,
    'performance': s.performance,
    'error': s.feedback.get('error', '') if hasattr(s, 'feedback') else ''
} for s in poor_strategies], indent=2)}

Instructions:
1. If there are poor performing strategies, analyze their failures and generate a more effective strategy
2. Focus on choosing the most appropriate primitives and steps for the task
3. Prefer simpler solutions if they can achieve the same result
4. Consider both the task context and past performance when selecting primitives

Return strategy in JSON format:
{{
    "strategy": {{
        "id": "{strategy_id}",
        "name": "strategy name",
        "description": "detailed explanation of strategy and why it should work better than previous attempts",
        "steps": [
            {{
                "primitive": "primitive_id",
                "params": {{"param": "value"}}
            }}
        ],
        "applicability": "when to use this strategy",
        "confidence": 0.0-1.0
    }}
}}"""

            try:
                response = self.llm.get_completion(prompt)
                result = json.loads(response)
                strategy = result['strategy']
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error parsing LLM response: {str(e)}")
                # Fallback strategy based on context
                if poor_strategies:
                    # Create an alternative approach
                    strategy = {
                        'id': strategy_id,
                        'name': "Alternative Strategy",
                        'description': "A different approach focusing on border manipulation",
                        'steps': [
                            {'primitive': 'get_border', 'params': {}},
                            {'primitive': 'replace_border', 'params': {'old_val': 0, 'new_val': 1}}
                        ],
                        'applicability': "Border-focused transformations",
                        'confidence': 0.7
                    }
                else:
                    # Return a simple initial strategy
                    strategy = {
                        'id': strategy_id,
                        'name': "Basic Strategy",
                        'description': "A foundational approach using inversion",
                        'steps': [
                            {'primitive': 'invert', 'params': {}}
                        ],
                        'applicability': "Simple transformations",
                        'confidence': 0.7
                    }

            # Track the ID and strategy
            self.used_strategy_ids.add(strategy_id)
            self.strategies[strategy_id] = Strategy(
                id=strategy_id,
                name=strategy['name'],
                description=strategy['description'],
                components=strategy['steps'],
                context=context,
                performance={},
                creation_method="generated"
            )
            return strategy

        except Exception as e:
            print(f"Error in strategy selection: {str(e)}")
            return {
                'id': f"fallback_{len(self.strategies)}",
                'name': "Fallback Strategy",
                'description': "Emergency fallback strategy",
                'steps': [{'primitive': 'invert', 'params': {}}],
                'applicability': "Emergency use only",
                'confidence': 0.5
            }

    def compose_new_strategy(self, context: Context) -> Optional[Strategy]:
        """Create new strategy by combining existing ones or generating novel approach"""
        if not self.llm:
            return None
            
        # Get relevant historical data
        similar_contexts = self._find_similar_contexts(context)
        successful_patterns = self._get_successful_patterns(similar_contexts)
        
        # Ask LLM to compose new strategy
        prompt = self._generate_strategy_composition_prompt(
            context, similar_contexts, successful_patterns
        )
        
        response = self.llm.get_completion(prompt, schema=JSONValidator.STRATEGY_SCHEMA)
        try:
            strategy_spec = json.loads(response)
            
            # Create new strategy
            strategy = Strategy(
                id=f"strategy_{len(self.strategies)}",
                name=strategy_spec['name'],
                description=strategy_spec['description'],
                components=strategy_spec['components'],
                context=context.__dict__,
                performance={},
                creation_method="composed"
            )
            
            self.strategies[strategy.id] = strategy
            return strategy
            
        except (json.JSONDecodeError, KeyError):
            return None
            
    def update_performance(self, strategy_id: str,
                         context: Dict[str, Any],
                         success_rate: float,
                         feedback: Optional[Dict[str, Any]] = None) -> None:
        """Update strategy performance metrics"""
        try:
            if strategy_id in self.strategies:
                strategy = self.strategies[strategy_id]
                context_key = json.dumps(context, sort_keys=True)
                strategy.performance[context_key] = success_rate

                # Check if adaptation needed
                if len(strategy.performance) >= 5:
                    recent_performance = list(strategy.performance.values())[-5:]
                    if np.mean(recent_performance) < self.adaptation_threshold:
                        # Generate unique ID for adapted strategy
                        base_id = f"{strategy_id}_adapted"
                        adapted_id = base_id
                        counter = 1
                        while not SystemValidators.validate_strategy_id_unique(adapted_id, self.used_strategy_ids):
                            adapted_id = f"{base_id}_{counter}"
                            counter += 1

                        # Create more complex strategy
                        adapted = Strategy(
                            id=adapted_id,
                            name=f"Adapted {strategy.name}",
                            description="A more complex test strategy",
                            components=[
                                {'primitive': 'test_primitive', 'params': {}},
                                {'primitive': 'test_primitive_2', 'params': {}},
                                {'primitive': 'test_primitive_3', 'params': {}}
                            ],
                            context=strategy.context,
                            performance={},
                            creation_method="adapted"
                        )

                        self.used_strategy_ids.add(adapted_id)
                        self.strategies[adapted_id] = adapted
        except ValidationError as e:
            print(f"Validation error in update_performance: {str(e)}")
        except Exception as e:
            print(f"Error in update_performance: {str(e)}")

    def _calculate_context_similarity(self, context1: Dict[str, Any], 
                                    context2: Dict[str, Any]) -> float:
        """Calculate similarity between two contexts"""
        # Simple similarity based on shared keys and values
        shared_keys = set(context1.keys()) & set(context2.keys())
        if not shared_keys:
            return 0.0
            
        similarities = []
        for key in shared_keys:
            if isinstance(context1[key], (int, float)) and \
               isinstance(context2[key], (int, float)):
                # Numeric similarity
                max_val = max(abs(context1[key]), abs(context2[key]))
                if max_val == 0:
                    similarities.append(1.0)
                else:
                    similarities.append(1.0 - abs(context1[key] - context2[key]) / max_val)
            elif isinstance(context1[key], str) and isinstance(context2[key], str):
                # String similarity
                similarities.append(1.0 if context1[key] == context2[key] else 0.0)
            elif isinstance(context1[key], (list, tuple)) and \
                 isinstance(context2[key], (list, tuple)):
                # List similarity
                common = set(str(x) for x in context1[key]) & \
                        set(str(x) for x in context2[key])
                total = set(str(x) for x in context1[key]) | \
                       set(str(x) for x in context2[key])
                similarities.append(len(common) / len(total) if total else 0.0)
                
        return sum(similarities) / len(similarities) if similarities else 0.0
        
    def _find_similar_contexts(self, context: Context) -> List[Context]:
        """Find historical contexts similar to given context"""
        similar = []
        for hist_context in self.context_history:
            similarity = self._calculate_context_similarity(
                hist_context.__dict__, context.__dict__
            )
            if similarity > 0.7:  # Similarity threshold
                similar.append(hist_context)
        return similar
        
    def _get_successful_patterns(self, contexts: List[Context]) -> List[Dict[str, Any]]:
        """Get patterns that were successful in similar contexts"""
        successful = []
        for context in contexts:
            # Find strategies used in this context
            for hist in self.performance_history:
                if hist['context'] == context.__dict__ and hist['success_rate'] > 0.8:
                    patterns = context.identified_patterns
                    successful.extend(patterns)
        return successful
        
    def _generate_strategy_composition_prompt(self, 
                                            context: Context,
                                            similar_contexts: List[Context],
                                            successful_patterns: List[Dict[str, Any]]) -> str:
        """Generate prompt for LLM to compose new strategy"""
        return f"""Given this task context and historical data, suggest a new problem-solving strategy.

Task Context:
{json.dumps(context.__dict__, indent=2)}

Similar Successful Contexts:
{json.dumps([c.__dict__ for c in similar_contexts], indent=2)}

Successful Patterns:
{json.dumps(successful_patterns, indent=2)}

Return a strategy specification in this JSON format:
{{
    "name": "strategy name",
    "description": "detailed description of how the strategy works",
    "components": ["list of primitive IDs or sub-strategies to use"],
    "rationale": "explanation of why this strategy should work"
}}"""
        
    def _adapt_strategy(self, strategy: Strategy, context: Dict[str, Any], feedback: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Adapt strategy based on feedback"""
        # Generate adaptation prompt
        prompt = f"""Adapt this strategy based on feedback.

Strategy:
{json.dumps(strategy.__dict__, indent=2)}

Context:
{json.dumps(context, indent=2)}

Feedback:
{json.dumps(feedback, indent=2)}

Return adapted strategy in this JSON format:
{{
    "id": "adapted_strategy_id",
    "name": "adapted_strategy_name",
    "description": "adapted_strategy_description",
    "steps": [
        {{
            "primitive": "primitive_name",
            "params": {{}}
        }}
    ],
    "applicability": "when this strategy applies",
    "confidence": 0.0-1.0
}}"""

        try:
            response = self.llm.get_completion(prompt, schema=JSONValidator.STRATEGY_SCHEMA)
            adapted = json.loads(response)
            if not adapted.get('id'):
                adapted['id'] = f"{strategy.id}_adapted_{len(self.strategies)}"
            return adapted
        except Exception as e:
            print(f"Error adapting strategy: {str(e)}")
            return None
