import json
import logging
from typing import Dict, Any, List, Optional, Set
from ..llm.llm_interface import LLMInterface
from ..utils.json_validator import JSONValidator
import numpy as np
import traceback

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
    
    # Add schema for solution validation
    SOLUTION_SCHEMA = {
        "type": "object",
        "required": ["steps"],
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["primitive", "params", "explanation"],
                    "properties": {
                        "primitive": {"type": "string"},
                        "params": {"type": "object"},
                        "explanation": {"type": "string"}
                    }
                }
            }
        }
    }

    STRATEGY_SCHEMA = {
        "type": "object",
        "required": ["id", "name", "description", "steps", "confidence", "explanation"],
        "properties": {
            "id": {"type": "string"},
            "name": {"type": "string", "minLength": 1},
            "description": {"type": "string", "minLength": 10},
            "steps": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": ["type", "description", "parameters"],
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": [
                                "invert",
                                "get_border",
                                "replace_border",
                                "rotate90",
                                "mirror",
                                "fill",
                                "get_interior",
                                "flood_fill",
                                "replace_color"
                            ]
                        },
                        "description": {"type": "string", "minLength": 10},
                        "parameters": {
                            "type": "object",
                            "allOf": [
                                {
                                    "if": {
                                        "properties": {"type": {"const": "invert"}},
                                        "required": ["type"]
                                    },
                                    "then": {
                                        "maxProperties": 0
                                    }
                                },
                                {
                                    "if": {
                                        "properties": {"type": {"const": "get_border"}},
                                        "required": ["type"]
                                    },
                                    "then": {
                                        "maxProperties": 0
                                    }
                                },
                                {
                                    "if": {
                                        "properties": {"type": {"const": "replace_border"}},
                                        "required": ["type"]
                                    },
                                    "then": {
                                        "required": ["old_val", "new_val"],
                                        "properties": {
                                            "old_val": {"type": "integer"},
                                            "new_val": {"type": "integer"}
                                        }
                                    }
                                },
                                {
                                    "if": {
                                        "properties": {"type": {"const": "rotate90"}},
                                        "required": ["type"]
                                    },
                                    "then": {
                                        "maxProperties": 0
                                    }
                                },
                                {
                                    "if": {
                                        "properties": {"type": {"const": "mirror"}},
                                        "required": ["type"]
                                    },
                                    "then": {
                                        "maxProperties": 0
                                    }
                                },
                                {
                                    "if": {
                                        "properties": {"type": {"const": "fill"}},
                                        "required": ["type"]
                                    },
                                    "then": {
                                        "required": ["value"],
                                        "properties": {
                                            "value": {"type": "integer"}
                                        }
                                    }
                                },
                                {
                                    "if": {
                                        "properties": {"type": {"const": "get_interior"}},
                                        "required": ["type"]
                                    },
                                    "then": {
                                        "maxProperties": 0
                                    }
                                },
                                {
                                    "if": {
                                        "properties": {"type": {"const": "flood_fill"}},
                                        "required": ["type"]
                                    },
                                    "then": {
                                        "required": ["x", "y", "value"],
                                        "properties": {
                                            "x": {"type": "integer"},
                                            "y": {"type": "integer"},
                                            "value": {"type": "integer"}
                                        }
                                    }
                                },
                                {
                                    "if": {
                                        "properties": {"type": {"const": "replace_color"}},
                                        "required": ["type"]
                                    },
                                    "then": {
                                        "required": ["old_val", "new_val"],
                                        "properties": {
                                            "old_val": {"type": "integer"},
                                            "new_val": {"type": "integer"}
                                        }
                                    }
                                }
                            ]
                        }
                    }
                }
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "explanation": {"type": "string", "minLength": 10}
        }
    }

    PATTERNS_SCHEMA = {
        "type": "object",
        "required": ["creation_patterns", "common_strategies", "meta_insights"],
        "properties": {
            "creation_patterns": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["pattern", "confidence"],
                    "properties": {
                        "pattern": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    }
                }
            },
            "common_strategies": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["strategy", "applicability"],
                    "properties": {
                        "strategy": {"type": "string"},
                        "applicability": {"type": "string"}
                    }
                }
            },
            "meta_insights": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["insight", "importance"],
                    "properties": {
                        "insight": {"type": "string"},
                        "importance": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        }
                    }
                }
            }
        }
    }

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
        self.primitives = {
            'invert': {
                'description': 'Invert the values in a grid (0->1, 1->0)',
                'parameters': {}
            },
            'get_border': {
                'description': 'Get the border cells of a grid',
                'parameters': {}
            },
            'replace_border': {
                'description': 'Replace border cells with new value',
                'parameters': {
                    'old_val': 'int',
                    'new_val': 'int'
                }
            },
            'rotate90': {
                'description': 'Rotate grid 90 degrees clockwise',
                'parameters': {}
            },
            'mirror': {
                'description': 'Mirror grid horizontally',
                'parameters': {}
            },
            'fill': {
                'description': 'Fill grid with value',
                'parameters': {
                    'value': 'int'
                }
            },
            'get_interior': {
                'description': 'Get interior (non-border) cells',
                'parameters': {}
            },
            'flood_fill': {
                'description': 'Fill connected region starting from point',
                'parameters': {
                    'x': 'int',
                    'y': 'int',
                    'value': 'int'
                }
            },
            'replace_color': {
                'description': 'Replace one color with another',
                'parameters': {
                    'old_val': 'int',
                    'new_val': 'int'
                }
            }
        }
        logger.info("MetaStrategyEngine initialized")

    async def learn_strategy_creation(self, examples: List[Dict[str, Any]], 
                                    relevant_patterns: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Learn patterns in strategy creation from examples."""
        logger.info("Starting learn_strategy_creation...")
        
        try:
            # Generate strategies for examples
            logger.info("Generating strategies for examples...")
            strategies = []
            for example in examples:
                input_grid = example['input']
                output_grid = example['output']
                strategy = await self.generate_strategy(input_grid, output_grid)
                if strategy:
                    strategies.append(strategy)
            
            if not strategies:
                logger.error("No valid strategies generated")
                return {
                    'creation_patterns': [],
                    'common_strategies': [],
                    'meta_insights': [],
                    'success': False
                }
            
            # Analyze patterns in successful strategies
            patterns_prompt = f"""Analyze these strategies and identify patterns in their creation and application.
Return your analysis in the following JSON format:
{{
    "creation_patterns": [
        {{
            "pattern": "description",
            "confidence": 0.0-1.0
        }}
    ],
    "common_strategies": [
        {{
            "strategy": "description",
            "applicability": "when to use"
        }}
    ],
    "meta_insights": [
        {{
            "insight": "description",
            "importance": 0.0-1.0
        }}
    ]
}}

Strategies to analyze:
{json.dumps(strategies, indent=2)}"""

            patterns_response = await self.llm.get_completion(patterns_prompt, schema=self.PATTERNS_SCHEMA)
            patterns_result = json.loads(patterns_response)
            
            # Add success indicator
            patterns_result['success'] = True
            
            return patterns_result
        except Exception as e:
            logger.error(f"Error analyzing strategy patterns: {str(e)}")
            return {
                'creation_patterns': [],
                'common_strategies': [],
                'meta_insights': [],
                'success': False
            }
            
    async def generate_strategy(self, input_grid: List[List[int]], output_grid: List[List[int]]) -> Dict[str, Any]:
        """Generate a strategy to transform input grid into output grid."""
        logger.info("Starting generate_strategy...")
        
        try:
            strategy_id = f"strategy_{len(self.strategies)}"
            self.used_strategy_ids.add(strategy_id)
            
            # Create strategy prompt with explicit JSON format and primitive details
            primitives_info = []
            for name, info in self.primitives.items():
                if info['parameters']:
                    param_info = [f"  {param}: {ptype} (REQUIRED)" for param, ptype in info['parameters'].items()]
                    primitives_info.append(f"{name}: {info['description']}\n  Required Parameters:\n" + "\n".join(param_info))
                else:
                    primitives_info.append(f"{name}: {info['description']}\n  No parameters required")
            
            prompt = f"""Analyze these grids and generate a strategy to transform the input into the output.
Return your response in the following JSON format:
{{
    "id": "{strategy_id}",
    "name": "strategy name",
    "description": "detailed description of the overall strategy",
    "steps": [
        {{
            "type": "primitive_name",
            "description": "Clear description of what this step does",
            "parameters": {{
                # Parameters MUST match primitive requirements exactly
                # Empty dict {{}} for primitives with no parameters
                # All required parameters for primitives that need them
            }}
        }}
    ],
    "confidence": 0.0-1.0,
    "explanation": "why this strategy should work"
}}

Available primitives and their parameter requirements:
{chr(10).join(primitives_info)}

Input Grid:
{self._format_grid(input_grid)}

Output Grid:
{self._format_grid(output_grid)}

Requirements:
1. Response MUST be valid JSON
2. Each step must use an available primitive exactly as shown above
3. Parameters must match the primitive's requirements exactly:
   - Use empty dict {{}} for primitives that take no parameters
   - Include ALL required parameters for primitives that need them
   - Parameters must be integers (not strings or floats)
4. Description must clearly explain the transformation
5. Each step must have a valid primitive type from the list above"""

            logger.info("Generated prompt:")
            logger.info(prompt)

            # Get strategy with schema validation
            response = await self.llm.get_completion(prompt, schema=self.STRATEGY_SCHEMA)
            if not response:
                logger.error("Empty response from LLM")
                return {
                    "id": strategy_id,
                    "name": "Failed Strategy",
                    "description": "Failed to generate strategy",
                    "steps": [],
                    "confidence": 0.0,
                    "explanation": "Failed to get LLM response"
                }
            
            try:
                strategy = json.loads(response)
                logger.info(f"Received strategy response:")
                logger.info(json.dumps(strategy, indent=2))
                
                # Additional validation of primitive parameters
                for i, step in enumerate(strategy.get('steps', [])):
                    primitive_type = step.get('type')
                    if primitive_type not in self.primitives:
                        error = f"Unknown primitive type: {primitive_type}"
                        logger.error(error)
                        raise ValueError(error)
                        
                    required_params = self.primitives[primitive_type]['parameters']
                    step_params = step.get('parameters', {})
                    
                    # Log step validation
                    logger.info(f"Validating step {i}:")
                    logger.info(f"  Type: {primitive_type}")
                    logger.info(f"  Required params: {required_params}")
                    logger.info(f"  Provided params: {step_params}")
                    
                    # Check for missing required parameters
                    for param, ptype in required_params.items():
                        if param not in step_params:
                            error = f"Missing required parameter '{param}' for primitive '{primitive_type}'"
                            logger.error(error)
                            raise ValueError(error)
                            
                    # Check for unexpected parameters
                    for param in step_params:
                        if param not in required_params:
                            error = f"Unexpected parameter '{param}' for primitive '{primitive_type}'"
                            logger.error(error)
                            raise ValueError(error)
                
                logger.info(f"Generated strategy: {strategy.get('name', 'Unnamed')}")
                return strategy
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse strategy response: {str(e)}")
                logger.error(f"Raw response: {response}")
                return {
                    "id": strategy_id,
                    "name": "Failed Strategy",
                    "description": "Failed to parse strategy",
                    "steps": [],
                    "confidence": 0.0,
                    "explanation": f"JSON parse error: {str(e)}"
                }
                
        except Exception as e:
            logger.error(f"Error generating strategy: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            return {
                "id": strategy_id,
                "name": "Failed Strategy", 
                "description": "Error generating strategy",
                "steps": [],
                "confidence": 0.0,
                "explanation": f"Error: {str(e)}"
            }

    def _format_grid(self, grid: List[List[int]]) -> str:
        """Format grid for prompt."""
        return '\n'.join(' '.join(str(x) for x in row) for row in grid)
        
    async def analyze_patterns(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze patterns in task examples."""
        try:
            logger.info("Analyzing transformation pattern...")
            
            if 'train' not in task:
                logger.error("Error: No training examples found")
                return [{
                    "type": "error",
                    "description": "No training examples found",
                    "rules": [],
                    "confidence": 0.0
                }]
                
            # Format task examples for analysis
            logger.info("Processing training examples...")
            examples_str = ""
            for i, example in enumerate(task['train'], 1):
                examples_str += f"\nExample {i}:\n"
                examples_str += f"Input:\n{self._format_grid(example['input'])}\n"
                examples_str += f"Output:\n{self._format_grid(example['output'])}\n"
            
            # Create analysis prompt
            logger.info("Requesting GPT-4 analysis...")
            prompt = f"""
            Analyze these transformation examples and identify the pattern:
            
            {examples_str}
            
            Identify:
            1. What changes from input to output
            2. How the pattern is consistent across examples
            3. The specific steps needed to transform input to output
            
            Format your response as a JSON array like this:
            [
                {{
                    "type": "transformation",
                    "description": "Detailed description of what changes",
                    "rules": [
                        "Identify rows with non-zero elements",
                        "Move these rows up by one position",
                        "Fill empty spaces with zeros"
                    ],
                    "confidence": 0.95,
                    "implementation": {{
                        "steps": [
                            {{
                                "type": "identify",
                                "description": "Find rows with non-zero elements"
                            }},
                            {{
                                "type": "transform",
                                "description": "Move rows up by one position"
                            }},
                            {{
                                "type": "fill",
                                "description": "Fill remaining spaces with zeros"
                            }}
                        ]
                    }}
                }}
            ]
            """
            
            # Get LLM analysis
            response = await self.llm.get_completion(prompt)
            
            # Try to parse as JSON first
            try:
                logger.info("Parsing pattern analysis...")
                import json
                patterns = json.loads(response)
                if isinstance(patterns, list) and patterns:
                    logger.info("Successfully identified pattern structure")
                    return patterns
            except:
                logger.error("Could not parse JSON response, creating structured pattern from text...")
                # If JSON parsing fails, create a structured pattern from the text
                return [{
                    "type": "transformation",
                    "description": response.strip(),
                    "rules": [
                        "Move non-zero elements up",
                        "Preserve element order",
                        "Fill empty spaces with zeros"
                    ],
                    "confidence": 0.8,
                    "implementation": {
                        "steps": [
                            {
                                "type": "identify",
                                "description": "Find non-zero elements"
                            },
                            {
                                "type": "transform",
                                "description": "Move elements up"
                            },
                            {
                                "type": "fill",
                                "description": "Fill with zeros"
                            }
                        ]
                    }
                }]
                
        except Exception as e:
            logger.error(f"\nError analyzing patterns: {type(e).__name__}: {str(e)}")
            return [{
                "type": "error",
                "description": str(e),
                "rules": [],
                "confidence": 0.0,
                "implementation": {"steps": []}
            }]
            
    async def generate_strategy(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a strategy based on identified patterns."""
        try:
            logger.info("Generating solution strategy...")
            
            if not patterns:
                logger.error("Error: No patterns provided")
                return {
                    "name": "error",
                    "description": "No patterns provided",
                    "steps": [],
                    "validation": {"checks": [], "threshold": 0.0}
                }
                
            # Extract implementation steps from patterns
            logger.info("Extracting implementation steps...")
            steps = []
            for pattern in patterns:
                if 'implementation' in pattern:
                    steps.extend(pattern['implementation'].get('steps', []))
                    
            # Create strategy
            logger.info("Creating strategy structure...")
            strategy = {
                "name": "pattern_transformation",
                "description": patterns[0].get('description', 'No description available'),
                "steps": steps,
                "validation": {
                    "checks": [
                        "Verify non-zero elements are moved correctly",
                        "Verify order is preserved",
                        "Verify zeros are filled correctly"
                    ],
                    "threshold": patterns[0].get('confidence', 0.8)
                }
            }
            
            logger.info("Strategy generation complete")
            return strategy
            
        except Exception as e:
            logger.error(f"\nError generating strategy: {type(e).__name__}: {str(e)}")
            return {
                "name": "error",
                "description": f"Error generating strategy: {str(e)}",
                "steps": [],
                "validation": {"checks": [], "threshold": 0.0}
            }
            
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

    async def generate_solution(self, insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate solution steps based on insights and learned strategies."""
        try:
            # Create prompt for solution generation
            prompt = f"""Based on these insights, generate a solution using available primitives:

Insights:
{json.dumps(insights, indent=2)}

Available Primitives:
{json.dumps(self.primitives, indent=2)}

Return a list of solution steps in this format:
{{
    "steps": [
        {{
            "primitive": "primitive_name",
            "params": {{"param1": "value1"}},
            "explanation": "why this step is needed"
        }}
    ]
}}"""

            if prompt is None:
                logger.error("Prompt is None")
                return []

            # Get solution from LLM
            response = await self.llm.get_completion(prompt, schema=self.SOLUTION_SCHEMA)
            if not response:
                logger.error("Empty response from LLM")
                return []
                
            try:
                parsed = json.loads(response)
                return parsed.get('steps', [])
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse solution response: {str(e)}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating solution: {str(e)}")
            return []

    async def analyze_patterns_new(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze patterns in task examples."""
        try:
            # Format task examples
            examples = []
            if 'train' in task:
                for example in task['train']:
                    examples.append({
                        'input': self._format_grid(example['input']),
                        'output': self._format_grid(example['output'])
                    })
            
            # Create analysis prompt
            prompt = f"""
            Analyze these transformation examples and identify the pattern:
            
            Training Examples:
            {examples}
            
            Identify:
            1. The core transformation pattern
            2. Any consistent rules or constraints
            3. Edge cases or special conditions
            
            Return a JSON array of patterns with this structure:
            [
                {{
                    "type": "transformation_pattern",
                    "description": "Detailed description of the pattern",
                    "rules": ["list", "of", "rules"],
                    "confidence": 0.95
                }},
                {{
                    "type": "constraint",
                    "description": "Description of any constraints",
                    "rules": ["list", "of", "constraints"],
                    "confidence": 0.9
                }}
            ]
            """
            
            # Get LLM analysis
            response = await self.llm.get_completion(prompt)
            
            # Parse response
            try:
                import json
                patterns = json.loads(response)
                if isinstance(patterns, list) and patterns:
                    return patterns
            except:
                pass
            
            # Fallback pattern if parsing fails
            return [{
                "type": "transformation_pattern",
                "description": response if response else "Pattern analysis failed",
                "rules": [],
                "confidence": 0.5
            }]
            
        except Exception as e:
            print(f"Error analyzing patterns: {str(e)}")
            return [{
                "type": "error",
                "description": f"Error analyzing patterns: {str(e)}",
                "rules": [],
                "confidence": 0.0
            }]

    async def generate_strategy_new(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a strategy based on identified patterns."""
        try:
            # Create strategy prompt
            prompt = f"""
            Generate a solution strategy based on these patterns:
            {patterns}
            
            Return a JSON object with this structure:
            {{
                "name": "strategy_name",
                "description": "strategy_description",
                "steps": [
                    {{
                        "type": "step_type",
                        "description": "step_description",
                        "parameters": {{"param": "value"}}
                    }}
                ],
                "validation": {{
                    "checks": ["list", "of", "validation", "checks"],
                    "threshold": 0.9
                }}
            }}
            """
            
            # Get LLM strategy
            response = await self.llm.get_completion(prompt)
            
            # Parse response
            try:
                import json
                return json.loads(response)
            except:
                return {
                    "name": "basic_strategy",
                    "description": response if response else "Strategy generation failed",
                    "steps": [],
                    "validation": {"checks": [], "threshold": 0.5}
                }
                
        except Exception as e:
            print(f"Error generating strategy: {str(e)}")
            return {
                "name": "error",
                "description": f"Error generating strategy: {str(e)}",
                "steps": [],
                "validation": {"checks": [], "threshold": 0.0}
            }
