from typing import Dict, Any, List
from openai import OpenAI
from dataclasses import dataclass
import json

@dataclass
class LLMResponse:
    text: str
    confidence: float

class LLMInterface:
    def __init__(self, api_key: str):
        """Initialize LLM interface with OpenAI API key."""
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4"
        
    def analyze_pattern(self, task: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
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
                f"1. Analyze each example individually\n"
                f"2. Compare the examples to find common patterns\n"
                f"3. Identify a unified transformation strategy that works for all examples\n\n"
                f"Here are the examples:\n\n"
                f"{chr(10).join(examples_text)}\n"
                f"Please analyze these examples and provide your findings in this EXACT format:\n\n"
                f"OBJECT PATTERNS:\n"
                f"- description: [pattern description]\n"
                f"- confidence: [high/medium/low]\n"
                f"- consistency: [how consistent across examples]\n\n"
                f"TRANSFORMATION PATTERNS:\n"
                f"- description: [pattern description]\n"
                f"- confidence: [high/medium/low]\n"
                f"- consistency: [how consistent across examples]\n\n"
                f"RELATIONSHIP PATTERNS:\n"
                f"- description: [pattern description]\n"
                f"- confidence: [high/medium/low]\n"
                f"- consistency: [how consistent across examples]\n\n"
                f"ABSTRACT PATTERNS:\n"
                f"- description: [pattern description]\n"
                f"- confidence: [high/medium/low]\n"
                f"- consistency: [how consistent across examples]\n\n"
                f"UNIFIED STRATEGY:\n"
                f"[Describe the overall strategy that works for all examples]\n\n"
                f"Focus on patterns that are consistent across ALL examples."
            )
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing visual patterns and transformations in the ARC challenge. Focus on finding consistent patterns across multiple examples."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
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
            print(f"Error in pattern analysis: {str(e)}")
            return {
                'object': [],
                'transformation': [],
                'relationship': [],
                'abstract': []
            }
            
    def validate_concept(self, concept: str, context: Dict[str, Any]) -> LLMResponse:
        """Validate a concept using LLM."""
        try:
            prompt = self._create_validation_prompt(concept, context)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at validating visual transformation concepts in the ARC challenge."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
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
                    
            return LLMResponse(text=text, confidence=confidence)
            
        except Exception as e:
            print(f"Error in concept validation: {str(e)}")
            return LLMResponse(text="Error validating concept", confidence=0.0)
            
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
