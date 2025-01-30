from typing import Dict, Any, List, Optional
from openai import OpenAI
from dataclasses import dataclass
import json
from src.utils.json_validator import JSONValidator
import logging
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    text: str
    confidence: float
    is_valid: bool = True  # Default to True since we'll set it to False if validation fails

class LLMInterface:
    """Interface for interacting with LLM models"""
    
    def __init__(self, api_key: str):
        """Initialize LLM interface with OpenAI API key."""
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini-2024-07-18"  # Restored original model name
        logger.info(f"Initialized LLMInterface with model: {self.model}")
        
    async def analyze_pattern(self, task: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
        """Analyze patterns in the task and return identified concepts."""
        try:
            # Check if we have multiple training examples
            train_examples = task.get('train', [])
            if not train_examples:
                raise ValueError("No training examples provided")
                
            # Format all examples for analysis
            examples_text = []
            for i, example in enumerate(train_examples):
                input_grid = example.get('input')
                output_grid = example.get('output')
                if input_grid is None or output_grid is None:
                    raise ValueError(f"Example {i} missing input/output data")
                    
                examples_text.append(
                    f"Example {i + 1}:\n"
                    f"Input grid:\n{self._format_grid(input_grid)}\n"
                    f"Output grid:\n{self._format_grid(output_grid)}\n"
                )
            
            # Create a detailed analysis prompt that compares examples
            prompt = (
                f"You are an expert at analyzing visual transformation patterns in the ARC (Abstraction and Reasoning Corpus) challenge.\n\n"
                f"I will show you {len(train_examples)} training examples. Your task is to:\n"
                f"1. Analyze the input and output grids carefully, paying attention to how individual cell values (0s and 1s) change\n"
                f"2. Compare the examples to find consistent transformation rules\n"
                f"3. Identify a unified strategy that works for all examples\n\n"
                f"Here are the examples:\n\n"
                f"{chr(10).join(examples_text)}\n"
                f"Please analyze these examples and provide your findings in this EXACT format:\n\n"
                f"OBJECT PATTERNS:\n"
                f"- description: [Describe what you observe in the INPUT grids - focus on the arrangement of 0s and 1s]\n"
                f"- confidence: [high/medium/low]\n"
                f"- consistency: [how consistent across examples]\n\n"
                f"TRANSFORMATION PATTERNS:\n"
                f"- description: [Describe EXACTLY how each cell changes from input to output - which 0s become 1s and which 1s become 0s]\n"
                f"- confidence: [high/medium/low]\n"
                f"- consistency: [how consistent across examples]\n\n"
                f"RELATIONSHIP PATTERNS:\n"
                f"- description: [Describe the relationship between input and output values - what determines if a cell becomes 0 or 1]\n"
                f"- confidence: [high/medium/low]\n"
                f"- consistency: [how consistent across examples]\n\n"
                f"ABSTRACT PATTERNS:\n"
                f"- description: [Describe the high-level concept or principle behind the transformation]\n"
                f"- confidence: [high/medium/low]\n"
                f"- consistency: [how consistent across examples]\n\n"
                f"UNIFIED STRATEGY:\n"
                f"[Describe step-by-step how to transform ANY input grid into its corresponding output. Be precise about which cells change and how.]\n\n"
                f"Focus on patterns that are consistent across ALL examples and be very precise about value changes."
            )
            
            print("\nMaking API call to OpenAI...")  # Show when we're actually making the call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing visual patterns and transformations in the ARC challenge. Focus on finding consistent patterns across multiple examples."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            print("Received response from OpenAI")  # Show when we get the response
            
            # Access the response content correctly
            raw_text = response.choices[0].message.content if response.choices else ""
            
            # Parse response into structured patterns by category
            patterns = {
                'object': [],
                'transformation': [],
                'relationship': [],
                'abstract': []
            }
            
            current_category = None
            current_pattern = None
            
            # Split response into sections
            sections = raw_text.split('\n\n')
            for section in sections:
                section = section.strip()
                
                # Skip empty sections
                if not section:
                    continue
                    
                # Check for category headers
                if section.upper().startswith(('OBJECT PATTERNS:', 'TRANSFORMATION PATTERNS:', 
                                            'RELATIONSHIP PATTERNS:', 'ABSTRACT PATTERNS:')):
                    header = section.split(':')[0].strip().upper()
                    if 'OBJECT' in header:
                        current_category = 'object'
                    elif 'TRANSFORMATION' in header:
                        current_category = 'transformation'
                    elif 'RELATIONSHIP' in header:
                        current_category = 'relationship'
                    elif 'ABSTRACT' in header:
                        current_category = 'abstract'
                        
                    # Extract patterns from this section
                    pattern_lines = section.split('\n')[1:]  # Skip header
                    current_pattern = {}
                    
                    for line in pattern_lines:
                        line = line.strip()
                        if line.startswith('- '):
                            key_value = line[2:].split(':', 1)
                            if len(key_value) == 2:
                                key = key_value[0].lower().strip()
                                value = key_value[1].strip()
                                
                                if key == 'confidence':
                                    # Parse confidence level
                                    if 'high' in value.lower():
                                        current_pattern[key] = 0.9
                                    elif 'medium' in value.lower():
                                        current_pattern[key] = 0.7
                                    elif 'low' in value.lower():
                                        current_pattern[key] = 0.5
                                else:
                                    current_pattern[key] = value
                                    
                    if current_pattern and current_category:
                        patterns[current_category].append(current_pattern)
                        
                # Check for unified strategy
                elif section.upper().startswith('UNIFIED STRATEGY:'):
                    strategy_text = section.split(':', 1)[1].strip()
                    patterns['abstract'].append({
                        'description': strategy_text,
                        'confidence': 0.9,
                        'consistency': 'Unified strategy for all examples',
                        'is_unified_strategy': True
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            return {
                'object': [],
                'transformation': [],
                'relationship': [],
                'abstract': []
            }
            
    async def validate_concept(self, concept: str, context: Dict[str, Any]) -> LLMResponse:
        """Validate a concept using LLM."""
        try:
            prompt = self._create_validation_prompt(concept, context)
            
            print("\nMaking API call to OpenAI...")  # Show when we're actually making the call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at validating visual transformation concepts in the ARC challenge."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            print("Received response from OpenAI")  # Show when we get the response
            
            text = response.choices[0].message.content if response.choices else ""
            confidence = 0.8  # Default confidence
            
            # Extract confidence from response
            if 'confidence:' in text.lower():
                conf_line = [line for line in text.split('\n') if 'confidence:' in line.lower()][0]
                conf_value = conf_line.split(':')[1].strip().lower()
                if 'high' in conf_value:
                    confidence = 0.9
                elif 'medium' in conf_value:
                    confidence = 0.7
                elif 'low' in conf_value:
                    confidence = 0.5
                    
            return LLMResponse(text=text, confidence=confidence, is_valid=True)
            
        except Exception as e:
            logger.error(f"Error in concept validation: {str(e)}")
            return LLMResponse(text="Error validating concept", confidence=0.0, is_valid=False)
            
    def _create_validation_prompt(self, concept: str, context: Dict[str, Any]) -> str:
        """Create prompt for concept validation."""
        return (
            f"Please validate the following concept in the context of an ARC task:\n\n"
            f"Concept: {concept}\n\n"
            f"Context:\n{json.dumps(context, indent=2)}\n\n"
            f"Please provide:\n"
            f"1. Whether the concept is valid and applicable\n"
            f"2. Your confidence level (High/Medium/Low)\n"
            f"3. Any potential issues or limitations\n"
            f"4. Suggestions for improvement"
        )
        
    def _format_grid(self, grid: List[List[int]]) -> str:
        """Format a grid for display in prompt."""
        return "\n".join(" ".join(str(cell) for cell in row) for row in grid)

    async def suggest_program_steps(self, pattern_info: Dict[str, str], 
                            available_primitives: Dict[str, str]) -> List[Dict[str, Any]]:
        """Ask LLM to suggest program steps based on patterns and available primitives."""
        
        # Build prompt with primitive descriptions and parameters
        primitives_info = []
        for name, desc in available_primitives.items():
            params = {
                'invert': {},
                'get_border': {},
                'replace_border': {'old_val': 'int', 'new_val': 'int'},
                'rotate90': {},
                'mirror': {},
                'fill': {'value': 'int'},
                'get_interior': {},
                'flood_fill': {'x': 'int', 'y': 'int', 'value': 'int'},
                'replace_color': {'old_val': 'int', 'new_val': 'int'},
                'extract_shape': {'value': 'int'},
                'apply_mask': {'mask': 'numpy array', 'value': 'int'},
                'find_pattern': {'pattern_value': 'int'},
                'get_pattern_bounds': {'pattern_value': 'int'},
                'expand_pattern': {'pattern': 'numpy array', 'expansion': 'int'}
            }.get(name, {})
            
            primitives_info.append(
                f"- {name}: {desc}\n"
                f"  Parameters: {json.dumps(params)}\n"
                f"  Note: Results from primitives are automatically stored and can be used by subsequent steps that need them."
            )
            
        prompt = (
            f"Given these patterns from analyzing a visual transformation task:\n\n"
            f"Object Pattern:\n{json.dumps(pattern_info.get('object', []), indent=2)}\n\n"
            f"Transformation Pattern:\n{json.dumps(pattern_info.get('transformation', []), indent=2)}\n\n" 
            f"Relationship Pattern:\n{json.dumps(pattern_info.get('relationship', []), indent=2)}\n\n"
            f"Abstract Pattern:\n{json.dumps(pattern_info.get('abstract', []), indent=2)}\n\n"
            f"\nAvailable Primitives:\n"
            + "\n".join(primitives_info) + "\n\n"
            f"Please analyze the patterns and suggest a sequence of primitive operations that would implement the transformation.\n"
            f"Important notes:\n"
            f"1. Focus on what the transformation is actually doing, not just matching the example\n"
            f"2. Consider different ways the transformation could be achieved with the available primitives\n"
            f"3. The system will automatically handle passing results between steps\n\n"
            f"For each step, specify:\n"
            f"1. Which primitive to use\n"
            f"2. What parameters to pass (must match the parameter names shown above)\n"
            f"3. Why this step is needed based on the patterns\n\n"
            f"Return your response in this exact JSON format with DOUBLE QUOTES:\n"
            f'{{\n'
            f'  "steps": [\n'
            f'    {{\n'
            f'      "primitive": "primitive_name",\n'
            f'      "params": {{"param1": value1}},\n'
            f'      "explanation": "Why this step is needed"\n'
            f'    }}\n'
            f'  ],\n'
            f'  "confidence": 0.9,\n'
            f'  "reasoning": "Explanation of overall approach"\n'
            f'}}\n\n'
            f'IMPORTANT:\n'
            f'1. Use double quotes (") for all JSON keys and string values, not single quotes\n'
            f'2. Do not add trailing commas after the last item in arrays or objects\n'
            f'3. Keep all text within the JSON structure - no additional text before or after\n'
            f'4. Make sure all strings are properly escaped\n'
        )
        
        print("\nMaking API call to OpenAI...")  # Show when we're actually making the call
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert at converting visual patterns into program steps. Think carefully about each step and ensure it uses the available primitives correctly. Always format JSON with double quotes."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        print("Received response from OpenAI")  # Show when we get the response
        
        # Parse response
        try:
            content = response.choices[0].message.content
            # Clean up the response to ensure valid JSON
            # Remove any trailing commas before closing braces/brackets
            content = content.replace(",}", "}")
            content = content.replace(",]", "]")
            
            # Replace any single quotes with double quotes
            content = content.replace("'", '"')
            
            # Try to extract just the JSON part if there's additional text
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                content = content[start:end]
            
            result = json.loads(content)
            return result.get('steps', [])
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return []

    async def get_completion(self, prompt: str, temperature: float = 0.7, schema: Optional[Dict[str, Any]] = None) -> str:
        """Get completion from LLM"""
        try:
            logger.info("Making API call to OpenAI...")
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            logger.info("Received response from OpenAI")
            
            if not response.choices:
                logger.error("No choices in OpenAI response")
                return ""
                
            text = response.choices[0].message.content
            logger.debug(f"Raw response text: {text[:100]}...")  # Log first 100 chars
            
            # Strip markdown code blocks if present
            if text.startswith("```"):
                lines = text.split("\n")
                # Remove first line (```json or similar)
                lines = lines[1:]
                # Find the closing ``` and remove everything after it
                for i, line in enumerate(lines):
                    if line.startswith("```"):
                        lines = lines[:i]
                        break
                text = "\n".join(lines)
                logger.debug(f"Stripped markdown blocks. New text: {text[:100]}...")
            
            if schema:
                logger.info("Validating response against schema...")
                is_valid, result, error = JSONValidator.validate_json(text, schema)
                if not is_valid:
                    logger.error(f"Response validation failed: {error}")
                    return ""
                logger.info("Response validation successful")
                return json.dumps(result)
            
            return text
            
        except Exception as e:
            logger.error(f"Error in get_completion: {str(e)}")
            return ""
