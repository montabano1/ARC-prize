from typing import Dict, Any, List, Optional
from openai import AsyncOpenAI
from dataclasses import dataclass
import json
import asyncio
import traceback
import os
import jsonschema
from datetime import datetime
import re

@dataclass
class LLMResponse:
    text: str
    confidence: float
    is_valid: bool = True  # Default to True since we'll set it to False if validation fails

class LLMInterface:
    """Interface for interacting with LLM."""
    
    # Define response schemas
    TRANSFORMATION_SCHEMA = {
        "type": "object",
        "required": ["steps"],
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["type", "description", "parameters"],
                    "properties": {
                        "type": {"type": "string"},
                        "description": {"type": "string"},
                        "parameters": {"type": "object"}
                    }
                }
            }
        }
    }

    ANALYSIS_SCHEMA = {
        "type": "object",
        "required": ["analysis"],
        "properties": {
            "analysis": {
                "type": "object",
                "required": ["shape_patterns", "color_patterns", "spatial_patterns", "transformation_rules"],
                "properties": {
                    "shape_patterns": {"type": "array", "items": {"type": "string"}},
                    "color_patterns": {"type": "array", "items": {"type": "string"}},
                    "spatial_patterns": {"type": "array", "items": {"type": "string"}},
                    "transformation_rules": {"type": "array", "items": {"type": "string"}}
                }
            }
        }
    }

    PROGRAM_SUGGESTION_SCHEMA = {
        "type": "object",
        "required": ["suggestions"],
        "properties": {
            "suggestions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["primitive_id", "confidence", "rationale"],
                    "properties": {
                        "primitive_id": {"type": "string"},
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0
                        },
                        "rationale": {"type": "string"}
                    }
                }
            }
        }
    }

    CONCEPT_VALIDATION_SCHEMA = {
        "type": "object",
        "required": ["validation"],
        "properties": {
            "validation": {
                "type": "object",
                "required": ["is_valid", "confidence", "issues", "suggestions"],
                "properties": {
                    "is_valid": {"type": "boolean"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "suggestions": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }
        }
    }

    STRATEGY_SCHEMA = {
        "type": "object",
        "required": ["steps"],
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "oneOf": [
                        {
                            "required": ["type", "description", "parameters"],
                            "properties": {
                                "type": {"type": "string"},
                                "description": {"type": "string"},
                                "parameters": {"type": "object"}
                            }
                        },
                        {
                            "required": ["primitive", "explanation", "params"],
                            "properties": {
                                "primitive": {"type": "string"},
                                "explanation": {"type": "string"},
                                "params": {"type": "object"}
                            }
                        }
                    ]
                }
            }
        }
    }

    HYPOTHESIS_SCHEMA = {
        "type": "object",
        "required": ["hypothesis", "confidence", "evidence", "test_cases"],
        "properties": {
            "hypothesis": {"type": "string"},
            "confidence": {"type": "number"},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "test_cases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["input", "expected_output", "explanation"],
                    "properties": {
                        "input": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                        "expected_output": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
                        "explanation": {"type": "string"}
                    }
                }
            }
        }
    }

    FEEDBACK_REQUEST_SCHEMA = {
        "type": "object",
        "required": ["query", "context", "options"],
        "properties": {
            "query": {"type": "string"},
            "context": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
            "importance": {"type": "number"}
        }
    }

    def __init__(self, api_key: Optional[str] = None):
        """Initialize LLM interface."""
        from openai import AsyncOpenAI
        
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("No OpenAI API key provided")
            
        self.client = AsyncOpenAI(api_key=self.api_key)
        self.model = "o3-mini"
        self.system_prompt = "You are an expert at analyzing and solving ARC tasks."
        
    async def analyze_pattern(self, task: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze patterns in the task and return identified concepts."""
        try:
            # Format the task data into a prompt
            prompt = (
                f"Analyze these patterns to discover concepts:\n\n"
                f"Input Patterns: {json.dumps(task.get('input_patterns', {}), indent=2)}\n"
                f"Output Patterns: {json.dumps(task.get('output_patterns', {}), indent=2)}\n"
                f"Previous Feedback: {json.dumps(task.get('feedback', 'null'), indent=2)}\n\n"
                "\nFocus on:\n"
                "1. Border patterns and relationships\n"
                "2. Transformation patterns\n"
                "3. Object patterns\n"
                "4. Pattern relationships\n\n"
                "Return concepts in JSON format:\n"
                "{\n"
                '    "concepts": [\n'
                "        {\n"
                '            "id": "concept_id",\n'
                '            "name": "concept name",\n'
                '            "description": "concept description",\n'
                '            "confidence": 0.95\n'
                "        }\n"
                "    ]\n"
                "}"
            )
            
            response = await self.get_completion(prompt, self.ANALYSIS_SCHEMA)
            print("\nChatGPT: Here's my analysis of the patterns:")
            return response
            
        except Exception as e:
            print(f"\nChatGPT: I encountered an issue while analyzing patterns: {str(e)}")
            return {"concepts": []}

    async def validate_concept(self, concept: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a concept using LLM."""
        try:
            prompt = self._create_validation_prompt(concept, context)
            response = await self.get_completion(prompt, self.CONCEPT_VALIDATION_SCHEMA)
            print("\nChatGPT: I've validated the concept. Here's what I found:")
            return response
            
        except Exception as e:
            print(f"\nChatGPT: I had trouble validating this concept: {str(e)}")
            return {"validation": {"is_valid": False, "confidence": 0.0, "issues": [str(e)], "suggestions": []}}

    async def suggest_program_steps(self, pattern_info: Dict[str, str], 
                                  available_primitives: Dict[str, str]) -> Dict[str, Any]:
        """Ask LLM to suggest program steps based on patterns and available primitives."""
        try:
            prompt = (
                f"Given these patterns:\n{json.dumps(pattern_info, indent=2)}\n\n"
                f"And these available primitives:\n{json.dumps(available_primitives, indent=2)}\n\n"
                "Suggest a sequence of program steps to implement the pattern transformation."
            )
            
            response = await self.get_completion(prompt, self.PROGRAM_SUGGESTION_SCHEMA)
            print("\nChatGPT: Here are my suggested program steps:")
            return response
            
        except Exception as e:
            print(f"\nChatGPT: I had trouble suggesting program steps: {str(e)}")
            return {"suggestions": []}

    async def analyze_context(self, context: Dict[str, Any]) -> str:
        """Analyze context and provide insights."""
        try:
            prompt = (
                f"Analyze this context and provide insights:\n{json.dumps(context, indent=2)}\n\n"
                "Focus on patterns, relationships, and potential strategies."
            )
            
            response = await self.get_completion(prompt)
            print("\nChatGPT: Based on the context, here's what I've found:")
            return response
            
        except Exception as e:
            print(f"\nChatGPT: I encountered an issue while analyzing context: {str(e)}")
            return ""

    async def explain_strategy(self, context: Dict[str, Any]) -> str:
        """Explain a solution strategy in clear terms."""
        try:
            prompt = f"""
            Analyze this task and explain the pattern:
            
            Training examples: {json.dumps(context['task']['train'], indent=2)}
            Test example: {json.dumps(context['task']['test'], indent=2)}
            
            Explain the pattern transformation in clear, step-by-step terms.
            Focus on:
            1. What changes from input to output
            2. How the pattern is consistent across examples
            3. The specific steps needed to transform input to output
            
            Return ONLY the explanation, without any prefixes or metadata.
            """
            
            response = await self.get_completion(prompt)
            
            # Clean up the response
            if response:
                # Remove any JSON formatting if present
                try:
                    parsed = json.loads(response)
                    if isinstance(parsed, dict) and 'explanation' in parsed:
                        return parsed['explanation']
                    elif isinstance(parsed, dict) and 'description' in parsed:
                        return parsed['description']
                except json.JSONDecodeError:
                    pass
                
                # Clean up the text response
                lines = response.split('\n')
                cleaned_lines = []
                for line in lines:
                    # Remove common prefixes
                    line = line.strip()
                    line = re.sub(r'^(Step \d+:|Pattern:|Rule:|•|\*|\-|1\.|2\.|3\.)\s*', '', line)
                    if line:
                        cleaned_lines.append(line)
                
                return '\n'.join(cleaned_lines)
            
            return ""
            
        except Exception as e:
            print(f"Error explaining strategy: {str(e)}")
            return ""

    async def request_feedback(self, hypothesis: str, confidence: float, context: str = "") -> Dict[str, Any]:
        """Request human feedback on a hypothesis."""
        try:
            prompt = self._create_feedback_prompt(hypothesis, confidence, context)
            response = await self.get_completion(prompt, self.FEEDBACK_REQUEST_SCHEMA)
            print("\nChatGPT: I need some clarification:")
            return response
            
        except Exception as e:
            print(f"\nChatGPT: I had trouble formulating my question: {str(e)}")
            return {
                "query": "I apologize, but I'm having trouble formulating my question clearly.",
                "context": context,
                "options": ["Continue", "Skip"],
                "importance": 0.5
            }

    async def propose_experiment(self, concept: Dict[str, Any]) -> Dict[str, Any]:
        """Propose an experiment to validate a concept."""
        try:
            prompt = (
                f"Design an experiment to validate this concept:\n"
                f"{json.dumps(concept, indent=2)}\n\n"
                "Include:\n"
                "1. Test cases\n"
                "2. Expected outcomes\n"
                "3. Success criteria"
            )
            
            response = await self.get_completion(prompt, self.HYPOTHESIS_SCHEMA)
            print("\nChatGPT: I propose the following experiment:")
            return response
            
        except Exception as e:
            print(f"\nChatGPT: I had trouble designing an experiment: {str(e)}")
            return {
                "hypothesis": "",
                "confidence": 0.0,
                "evidence": [],
                "test_cases": []
            }

    def _create_validation_prompt(self, concept: str, context: Dict[str, Any]) -> str:
        """Create prompt for concept validation."""
        return f"""Please validate the following concept in the context of an ARC task:

Concept: {concept}

Context:
{json.dumps(context, indent=2)}

Analyze the concept and provide your assessment in the following JSON format:
{{
    "validation": {{
        "is_valid": true/false,
        "confidence": 0.0-1.0,
        "issues": [
            "List any potential issues or limitations"
        ],
        "suggestions": [
            "List any suggestions for improvement"
        ]
    }}
}}

Consider:
1. Whether the concept accurately describes the transformation
2. If it can be implemented with available operations
3. Whether it generalizes beyond the given examples
4. Any edge cases or limitations

Your response MUST be valid JSON and follow the exact schema shown above.
"""

    def _format_grid(self, grid: List[List[int]]) -> str:
        """Format a grid for display in prompt."""
        # Convert grid to string representation with brackets
        rows = []
        for row in grid:
            row_str = "[" + " ".join(str(cell).rjust(2) for cell in row) + "]"
            rows.append(row_str)
        return "[\n  " + "\n  ".join(rows) + "\n]"

    def _generate_strategy_prompt(self, task: Dict[str, Any]) -> str:
        """Generate prompt for strategy creation."""
        return f"""Analyze these grids and generate a strategy to transform the input into the output.
Return your response in the following JSON format:
{{
    "steps": [
        {{
            "type": "primitive_name",  # Use the exact primitive name from below
            "description": "Clear description of what this step does",
            "parameters": {{}}  # Empty dict for primitives with no parameters, or include required parameters
        }}
    ]
}}

Available primitives and their parameter requirements:
invert: Invert the values in a grid (0->1, 1->0)
  No parameters required
get_border: Get the border cells of a grid
  No parameters required
replace_border: Replace border cells with new value
  Required Parameters:
  old_val: int (REQUIRED)
  new_val: int (REQUIRED)
rotate90: Rotate grid 90 degrees clockwise
  No parameters required
mirror: Mirror grid horizontally
  No parameters required
fill: Fill grid with value
  Required Parameters:
  value: int (REQUIRED)
get_interior: Get interior (non-border) cells
  No parameters required
flood_fill: Fill connected region starting from point
  Required Parameters:
  x: int (REQUIRED)
  y: int (REQUIRED)
  value: int (REQUIRED)
replace_color: Replace one color with another
  Required Parameters:
  old_val: int (REQUIRED)
  new_val: int (REQUIRED)
extract_shape: Extract shape from grid
  Required Parameters:
  value: int (REQUIRED)
apply_mask: Apply mask to grid
  Required Parameters:
  mask: list (REQUIRED)
  value: int (REQUIRED)
find_pattern: Find pattern in grid
  Required Parameters:
  pattern_value: int (REQUIRED)
get_pattern_bounds: Get bounds of pattern in grid
  Required Parameters:
  pattern_value: int (REQUIRED)
expand_pattern: Expand pattern in grid
  Required Parameters:
  pattern: list (REQUIRED)
  expansion: int (REQUIRED)

Input Grid:
{self._format_grid(task['input'])}

Output Grid:
{self._format_grid(task['output'])}

Requirements:
1. Response MUST be valid JSON
2. Each step must use an available primitive exactly as shown above
3. Parameters must match the primitive's requirements exactly:
   - Use empty dict {{}} for primitives that take no parameters
   - Include ALL required parameters for primitives that need them
   - Parameters must be integers (not strings or floats)
4. Description must clearly explain the transformation
5. Each step must have a valid primitive type from the list above
"""

    async def get_completion(self, prompt: str, max_retries: int = 3) -> str:
        """Get completion from LLM with retries."""
        for attempt in range(max_retries):
            try:
                print(f"\nRequesting GPT-4 analysis (attempt {attempt + 1}/{max_retries})...")
                
                response = await self._call_api(prompt)
                
                if hasattr(response, 'choices') and response.choices:
                    print("Successfully received GPT-4 response")
                    return response.choices[0].message.content
                    
                print("Error: Unexpected API response format")
                
            except Exception as e:
                error_type = type(e).__name__
                if attempt < max_retries - 1:
                    print(f"API call failed ({error_type}: {str(e)})")
                    print(f"Retrying in 1 second... (attempt {attempt + 2}/{max_retries})")
                    await asyncio.sleep(1)  # Add a small delay between retries
                else:
                    print(f"API call failed after {max_retries} attempts")
                    print(f"Final error: {error_type}: {str(e)}")
                    raise
                    
        raise Exception("Max retries reached without successful completion")

    async def _call_api(self, prompt: str, schema: Optional[Dict[str, Any]] = None) -> Any:
        """Make the actual API call."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
            )
            return response
        except Exception as e:
            print(f"API call failed: {str(e)}")
            raise

    def _check_response_quality(self, parsed_content: Dict[str, Any]) -> List[str]:
        """Check quality of parsed response content."""
        issues = []
        
        def check_field_quality(field: Any, field_name: str, min_length: int = 10):
            if isinstance(field, str):
                if len(field.strip()) < min_length:
                    issues.append(f"{field_name} is too short or empty")
                if field.strip().lower() in ['none', 'n/a', 'unknown']:
                    issues.append(f"{field_name} contains placeholder text")
            elif isinstance(field, list):
                if not field:
                    issues.append(f"{field_name} list is empty")
                else:
                    for i, item in enumerate(field):
                        check_field_quality(item, f"{field_name}[{i}]")
            elif isinstance(field, dict):
                if not field:
                    issues.append(f"{field_name} dictionary is empty")
                else:
                    for key, value in field.items():
                        check_field_quality(value, f"{field_name}.{key}")
        
        check_field_quality(parsed_content, "root")
        return issues

    def _validate_response(self, content: str, schema: Optional[Dict[str, Any]] = None) -> bool:
        """Validate response content and structure."""
        try:
            # Parse response
            parsed = json.loads(content)
            print("Successfully parsed response as JSON")
            print(f"Full response content:\n{json.dumps(parsed, indent=2)}")

            # Log root level fields
            print("Root level fields:")
            for key, value in parsed.items():
                if isinstance(value, dict):
                    print(f"  {key}: dict with {len(value)} items")
                elif isinstance(value, list):
                    print(f"  {key}: list with {len(value)} items")
                    # For steps, log each step's details
                    if key == "steps":
                        for i, step in enumerate(value):
                            print(f"    Step {i}:")
                            for step_key, step_value in step.items():
                                print(f"      {step_key}: {repr(step_value)}")
                else:
                    print(f"  {key}: {value}")

            # Validate steps if present
            if "steps" in parsed:
                print("Validating strategy steps...")
                for i, step in enumerate(parsed["steps"]):
                    print(f"Validating step {i}:")
                    
                    # Handle both "type" and "primitive" fields
                    step_type = step.get("type") or step.get("primitive", "")
                    params = step.get("parameters") or step.get("params", {})
                    desc = step.get("description") or step.get("explanation", "")
                    
                    print(f"  Type: {repr(step_type)}")
                    print(f"  Parameters: {params}")
                    print(f"  Description: {repr(desc)}")
                    
                    # Validate step
                    errors = []
                    if not step_type:
                        errors.append("root.steps[{}].type is invalid or empty".format(i))
                    if not desc or len(desc.strip()) < 10:  # Require meaningful descriptions
                        errors.append("root.steps[{}].description is too short or invalid".format(i))
                        
                    if errors:
                        print("Step validation failed:")
                        for error in errors:
                            print(f"- {error}")
                        return False

                print("All strategy steps validated successfully")
                return True

            return True

        except Exception as e:
            print(f"Response validation failed: {str(e)}")
            return False

    def _validate_transformation_response(self, response: Dict[str, Any]) -> None:
        """Validate transformation response schema with enhanced error checking."""
        try:
            if not isinstance(response, dict):
                raise ValueError("Response must be a dictionary")
                
            if "steps" not in response:
                raise ValueError("Response must contain 'steps' key")
                
            steps = response["steps"]
            if not isinstance(steps, list):
                raise ValueError("'steps' must be a list")
                
            if not steps:
                raise ValueError("'steps' list cannot be empty")
                
            for i, step in enumerate(steps):
                if not isinstance(step, dict):
                    raise ValueError(f"Step {i} must be a dictionary")
                    
                # Check required fields
                required_fields = ["type", "description", "parameters"]
                missing_fields = [f for f in required_fields if f not in step]
                if missing_fields:
                    raise ValueError(f"Step {i} missing required fields: {', '.join(missing_fields)}")
                    
                # Validate field types
                if not isinstance(step["type"], str):
                    raise ValueError(f"Step {i} 'type' must be a string")
                if not isinstance(step["description"], str):
                    raise ValueError(f"Step {i} 'description' must be a string")
                if not isinstance(step["parameters"], dict):
                    raise ValueError(f"Step {i} 'parameters' must be a dictionary")
                    
                # Validate field content quality
                if len(step["type"].strip()) < 2:
                    raise ValueError(f"Step {i} 'type' is too short")
                if len(step["description"].strip()) < 10:
                    raise ValueError(f"Step {i} 'description' is too brief")
                    
                # Check for placeholder text
                placeholder_values = ['none', 'n/a', 'unknown', 'todo', 'tbd']
                if step["type"].strip().lower() in placeholder_values:
                    raise ValueError(f"Step {i} 'type' contains placeholder text")
                if step["description"].strip().lower() in placeholder_values:
                    raise ValueError(f"Step {i} 'description' contains placeholder text")
                    
                # Validate parameters based on type
                self._validate_parameters(step["type"], step["parameters"], i)
                
        except ValueError as e:
            print(f"Validation error: {str(e)}")
            raise
            
    def _validate_parameters(self, step_type: str, parameters: Dict[str, Any], step_index: int) -> None:
        """Validate parameters for a specific step type."""
        # Define expected parameters for each step type
        parameter_specs = {
            "invert": {},
            "get_border": {},
            "replace_border": {
                "old_val": {"type": int, "required": True},
                "new_val": {"type": int, "required": True}
            },
            "rotate90": {},
            "mirror": {},
            "fill": {
                "value": {"type": int, "required": True}
            },
            "get_interior": {},
            "flood_fill": {
                "x": {"type": int, "required": True},
                "y": {"type": int, "required": True},
                "value": {"type": int, "required": True}
            },
            "replace_color": {
                "old_val": {"type": int, "required": True},
                "new_val": {"type": int, "required": True}
            },
            "extract_shape": {
                "value": {"type": int, "required": True}
            },
            "apply_mask": {
                "mask": {"type": list, "required": True},
                "value": {"type": int, "required": True}
            },
            "find_pattern": {
                "pattern_value": {"type": int, "required": True}
            },
            "get_pattern_bounds": {
                "pattern_value": {"type": int, "required": True}
            },
            "expand_pattern": {
                "pattern": {"type": list, "required": True},
                "expansion": {"type": int, "required": True}
            }
        }
        
        # Get parameter specification for this step type
        spec = parameter_specs.get(step_type)
        if spec is None:
            raise ValueError(f"Step {step_index} has unknown type: {step_type}")
            
        # Check for required parameters
        for param_name, param_spec in spec.items():
            if param_spec.get("required", False):
                if param_name not in parameters:
                    raise ValueError(f"Step {step_index} missing required parameter: {param_name}")
                    
                # Validate parameter type
                param_value = parameters[param_name]
                expected_type = param_spec["type"]
                if not isinstance(param_value, expected_type):
                    raise ValueError(f"Step {step_index} parameter '{param_name}' must be of type {expected_type.__name__}")
                    
                # Additional validation for specific types
                if expected_type == int:
                    if not isinstance(param_value, (int, float)) or isinstance(param_value, bool):
                        raise ValueError(f"Step {step_index} parameter '{param_name}' must be a number")
                elif expected_type == list:
                    if not param_value:
                        raise ValueError(f"Step {step_index} parameter '{param_name}' cannot be an empty list")
                        
        # Check for unexpected parameters
        for param_name in parameters:
            if param_name not in spec:
                raise ValueError(f"Step {step_index} has unexpected parameter: {param_name}")

    def _create_feedback_prompt(self, hypothesis: str, confidence: float, context: str) -> str:
        """Create a prompt for requesting feedback"""
        return f"""
        I have formed a hypothesis with {confidence:.2%} confidence:
        
        Hypothesis: {hypothesis}
        
        Context: {context}
        
        Please generate a feedback request in JSON format that:
        1. Asks specific questions about the hypothesis
        2. Provides context for why the feedback is important
        3. Suggests possible response options
        4. Indicates the importance of this feedback (0-1)
        
        Use this schema:
        {{
            "query": "Clear question about the hypothesis",
            "context": "Why this feedback is important",
            "options": ["Option 1", "Option 2", ...],
            "importance": 0.8
        }}
        """
