import json
import logging
from typing import Dict, Any, List, Optional, Set
from ..llm.llm_interface import LLMInterface
from ..utils.json_validator import JSONValidator
import numpy as np

logger = logging.getLogger(__name__)

class Strategy:
    def __init__(self, id: str, name: str, description: str, steps: List[Dict[str, Any]], 
                 applicability: str, confidence: float, context: Dict[str, Any]):
        self.id = id
        self.name = name
        self.description = description
        self.steps = steps
        self.applicability = applicability
        self.confidence = confidence
        self.context = context
        self.performance = []

class MetaStrategyEngine:
    """Learns how to create and evolve effective strategies"""
    
    def __init__(self, llm: LLMInterface):
        """Initialize meta strategy engine"""
        logger.info("Initializing MetaStrategyEngine...")
        self.llm = llm
        self.strategies: Dict[str, Strategy] = {}
        self.performance_history: Dict[str, List[float]] = {}
        self.context_history: List[Dict[str, Any]] = []
        self.adaptation_threshold = 0.7  # Performance threshold for adaptation
        self.used_strategy_ids: Set[str] = set()
        self.strategy_patterns: List[Dict[str, Any]] = []  # Patterns about what makes strategies successful
        logger.info("MetaStrategyEngine initialized")
        
    async def learn_strategy_creation(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn patterns about how to create effective strategies"""
        logger.info("Starting learn_strategy_creation...")
        
        # Generate strategies for each example individually
        logger.info("Generating strategies for examples...")
        strategies_by_example = []
        for example in examples:
            # Convert numpy arrays to lists
            if isinstance(example.get('input'), np.ndarray):
                example['input'] = example['input'].tolist()
            if isinstance(example.get('output'), np.ndarray):
                example['output'] = example['output'].tolist()
                
            context = self._extract_context(example)
            strategy = await self.generate_strategy(context)
            if strategy:
                # Store strategy in self.strategies
                strategy_obj = Strategy(
                    id=strategy['id'],
                    name=strategy['name'],
                    description=strategy['description'],
                    steps=strategy['steps'],
                    applicability=strategy['applicability'],
                    confidence=strategy['confidence'],
                    context=context
                )
                self.strategies[strategy['id']] = strategy_obj
                
                strategies_by_example.append({
                    'example': example,
                    'strategy': strategy,
                    'context': context
                })
        
        # Analyze what makes strategies successful
        logger.info("Analyzing strategy patterns...")
        prompt = f"""Analyze these strategies and identify patterns about strategy creation:

Strategies:
{json.dumps(strategies_by_example, indent=2)}

Return patterns in this format:
{{
    "creation_patterns": [
        {{
            "pattern": "description of pattern",
            "evidence": ["example1", "example2"],
            "confidence": 0.0-1.0
        }}
    ],
    "common_strategies": [
        {{
            "strategy": "strategy description",
            "applicability": "when to use",
            "confidence": 0.0-1.0
        }}
    ]
}}"""
        
        response = await self.llm.get_completion(prompt)
        try:
            patterns = json.loads(response)
            # Validate the response has the required fields
            if not isinstance(patterns, dict):
                logger.error("Strategy patterns response is not a dictionary")
                return {}
                
            if 'creation_patterns' not in patterns:
                logger.error("Strategy patterns missing creation_patterns")
                return {}
                
            if 'common_strategies' not in patterns:
                logger.error("Strategy patterns missing common_strategies")
                return {}
                
            # Validate each pattern
            for pattern in patterns['creation_patterns']:
                if not all(k in pattern for k in ('pattern', 'evidence', 'confidence')):
                    logger.error(f"Invalid pattern format: {pattern}")
                    continue
                    
            # Validate each strategy
            for strategy in patterns['common_strategies']:
                if not all(k in strategy for k in ('strategy', 'applicability', 'confidence')):
                    logger.error(f"Invalid strategy format: {strategy}")
                    continue
                    
            self.strategy_patterns.extend(patterns.get('creation_patterns', []))
            return patterns
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse strategy patterns: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Error processing strategy patterns: {str(e)}")
            return {}
            
    async def generate_strategy(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a new strategy using learned patterns"""
        logger.info("Starting generate_strategy...")
        strategy_id = f"strategy_{len(self.used_strategy_ids)}"
        
        # Convert numpy arrays to lists for JSON serialization
        if isinstance(context.get('input'), np.ndarray):
            context['input'] = context['input'].tolist()
        if isinstance(context.get('output'), np.ndarray):
            context['output'] = context['output'].tolist()
            
        # Build prompt using learned patterns
        patterns_str = ""
        if self.strategy_patterns:
            patterns_str = "\nLearned Strategy Patterns:\n" + json.dumps(self.strategy_patterns, indent=2)
            
        prompt = f"""Generate a strategy for solving this task:

Context:
{json.dumps(context, indent=2)}
{patterns_str}

Return strategy in this format:
{{
    "id": "{strategy_id}",
    "name": "strategy name",
    "description": "detailed strategy description",
    "steps": [
        {{
            "type": "step type",
            "description": "step description",
            "parameters": {{}}
        }}
    ],
    "applicability": "when to use this strategy",
    "confidence": 0.0-1.0
}}"""

        try:
            response = await self.llm.get_completion(prompt)
            strategy = json.loads(response)
            
            # Validate strategy format
            required_fields = ['id', 'name', 'description', 'steps', 'applicability', 'confidence']
            if not isinstance(strategy, dict):
                logger.error("Strategy response is not a dictionary")
                return None
                
            if not all(field in strategy for field in required_fields):
                logger.error(f"Strategy missing required fields. Has: {list(strategy.keys())}")
                return None
                
            # Validate steps
            for step in strategy['steps']:
                if not isinstance(step, dict) or not all(k in step for k in ('type', 'description')):
                    logger.error(f"Invalid step format: {step}")
                    return None
                    
            # Validate confidence is float between 0 and 1
            if not isinstance(strategy['confidence'], (int, float)) or not 0 <= strategy['confidence'] <= 1:
                logger.error(f"Invalid confidence value: {strategy['confidence']}")
                return None
                
            # Store strategy in self.strategies
            strategy_obj = Strategy(
                id=strategy['id'],
                name=strategy['name'],
                description=strategy['description'],
                steps=strategy['steps'],
                applicability=strategy['applicability'],
                confidence=strategy['confidence'],
                context=context
            )
            self.strategies[strategy['id']] = strategy_obj
            
            self.used_strategy_ids.add(strategy_id)
            return strategy
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse strategy: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            return None
        
    async def select_strategy(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Select best strategy for given context"""
        logger.info("Starting select_strategy...")
        
        # Get all strategies sorted by performance
        sorted_strategies = sorted(
            self.strategies.values(),
            key=lambda s: sum(s.performance) / len(s.performance) if s.performance else 0,
            reverse=True
        )
        
        if not sorted_strategies:
            logger.warning("No strategies available")
            return None
            
        # For now just return the best performing strategy
        best_strategy = sorted_strategies[0]
        return {
            'id': best_strategy.id,
            'name': best_strategy.name,
            'description': best_strategy.description,
            'steps': best_strategy.steps,
            'confidence': best_strategy.confidence
        }
        
    async def update_performance(self, strategy_id: str, accuracy: float, context: Dict[str, Any]) -> None:
        """Update performance metrics for a strategy"""
        if strategy_id not in self.strategies:
            logger.warning(f"Strategy {strategy_id} not found")
            return
            
        # Add performance to history
        if strategy_id not in self.performance_history:
            self.performance_history[strategy_id] = []
        self.performance_history[strategy_id].append(accuracy)
        
        # Store context
        self.context_history.append(context)
        
        # Update strategy performance
        self.strategies[strategy_id].performance.append(accuracy)
        
        logger.info(f"Updated performance for strategy {strategy_id}: {accuracy:.2f}")
        
    def _validate_strategy(self, strategy: Dict[str, Any]) -> bool:
        """Validate strategy has required fields"""
        required_fields = ['strategy_id', 'description', 'steps', 'applicability', 'confidence']
        return all(field in strategy for field in required_fields)
        
    def _extract_context(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context from example"""
        return {
            'input': example.get('input', []),
            'output': example.get('output', [])
        }
