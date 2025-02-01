from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
import numpy as np
from ..concept_formation.concept_extractor import ConceptExtractor
import json
from ..utils.json_validator import JSONValidator
from ..utils.validators import SystemValidators, ValidationError
from ..llm.llm_interface import LLMInterface
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class ConceptValidation:
    """Validation result for a concept"""
    is_valid: bool
    feedback: str

class PatternAbstractor:
    ABSTRACTION_SCHEMA = {
        "type": "object",
        "required": ["concepts", "confidence", "explanation"],
        "properties": {
            "concepts": {
                "type": "array",
                "items": {"type": "string"}
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "explanation": {"type": "string"}
        }
    }
    
    RELATIONSHIP_SCHEMA = {
        "type": "object",
        "required": ["relationship", "confidence", "explanation"],
        "properties": {
            "relationship": {"type": "string"},
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "explanation": {"type": "string"}
        }
    }
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.abstractions = {}
        
    async def abstract_patterns(self, patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Abstract higher-level patterns from observed patterns."""
        prompt = f"""Analyze these patterns and identify higher-level abstractions.
Return your analysis in the following JSON format:
{{
    "concepts": ["list of abstracted concepts"],
    "confidence": 0.0-1.0,
    "explanation": "why these abstractions make sense"
}}

Patterns to analyze:
{json.dumps(patterns, indent=2)}"""

        try:
            response = await self.llm.get_completion(prompt, schema=self.ABSTRACTION_SCHEMA)
            result = json.loads(response)
            
            # Store abstractions
            abstractions = {
                'concepts': result['concepts'],
                'confidence': result['confidence'],
                'relationships': await self._find_relationships(patterns)
            }
            
            return abstractions
            
        except Exception as e:
            logger.error(f"Error in pattern abstraction: {str(e)}")
            return {
                'concepts': [],
                'confidence': 0.0,
                'relationships': []
            }
        
    async def _find_relationships(self, patterns: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find relationships between patterns."""
        relationships = []
        for i, p1 in enumerate(patterns):
            for j, p2 in enumerate(patterns[i+1:], i+1):
                prompt = f"""Analyze the relationship between these two patterns.
Return your analysis in the following JSON format:
{{
    "relationship": "description of relationship",
    "confidence": 0.0-1.0,
    "explanation": "why this relationship exists"
}}

Pattern 1:
{json.dumps(p1, indent=2)}

Pattern 2:
{json.dumps(p2, indent=2)}"""
                
                try:
                    response = await self.llm.get_completion(prompt, schema=self.RELATIONSHIP_SCHEMA)
                    result = json.loads(response)
                    relationships.append(result)
                except Exception as e:
                    logger.error(f"Error finding relationship: {str(e)}")
                    continue
                        
        return relationships

class ConceptValidator:
    CONSISTENCY_SCHEMA = {
        "type": "object",
        "required": ["score", "explanation"],
        "properties": {
            "score": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "explanation": {"type": "string"}
        }
    }
    
    def __init__(self, llm: LLMInterface, confidence_threshold: float = 0.7):
        self.llm = llm
        self.validations = {}
        self.confidence_threshold = confidence_threshold
        
    async def validate_concept(self, concept: Dict[str, Any], 
                        examples: List[Dict[str, Any]]) -> ConceptValidation:
        """Validate a concept against examples."""
        # First try the optimized single-call validation
        try:
            prompt = f"""Analyze if this concept is valid across these examples:

Concept:
{json.dumps(concept, indent=2)}

Examples:
{json.dumps(examples, indent=2)}

Return response in this exact JSON format:
{{
    "generalization_score": 0.0-1.0,  # How well the concept generalizes across examples
    "consistency_score": 0.0-1.0,     # How consistent the concept is
    "supports_examples": [true/false], # List indicating if concept supports each example
    "explanation": "string"           # Explanation of the validation
}}"""

            response = await self.llm.get_completion(prompt, schema=JSONValidator.CONCEPT_VALIDATION_SCHEMA)
            result = json.loads(response)
            
            # Overall validation
            is_valid = (
                result['generalization_score'] > 0.7 and 
                result['consistency_score'] > 0.7 and
                all(result['supports_examples'])
            )
            
            validation = ConceptValidation(
                is_valid=is_valid,
                feedback=f"Generalization: {result['generalization_score']}, Consistency: {result['consistency_score']}\n{result['explanation']}"
            )
            
            return validation
            
        except Exception as e:
            logger.error(f"Error in optimized concept validation: {str(e)}, falling back to individual validation")
            # Fall back to individual validation methods if the optimized approach fails
            generalization_score = await self._test_generalization(concept, examples)
            consistency_score = await self._test_consistency(concept, examples)
            
            is_valid = generalization_score > 0.7 and consistency_score > 0.7
            
            return ConceptValidation(
                is_valid=is_valid,
                feedback=f"Generalization: {generalization_score}, Consistency: {consistency_score}"
            )
        
    async def _test_generalization(self, concept: Dict[str, Any], 
                           examples: List[Dict[str, Any]]) -> float:
        """Test how well the concept generalizes."""
        score = 0.0
        for example in examples:
            if await self._supports_concept(concept, example):
                score += 1.0
        return score / len(examples) if examples else 0.0
        
    async def _test_consistency(self, concept: Dict[str, Any], 
                         examples: List[Dict[str, Any]]) -> float:
        """Test consistency of concept across examples."""
        if not examples:
            return 0.0
            
        # Use LLM to check consistency
        prompt = f"""Check if this concept is consistent across examples.
Return your analysis in the following JSON format:
{{
    "score": 0.0-1.0,      # How consistent the concept is
    "explanation": "string" # Why you gave this score
}}

Concept:
{json.dumps(concept, indent=2)}

Examples:
{json.dumps(examples, indent=2)}"""

        try:
            response = await self.llm.get_completion(prompt, schema=self.CONSISTENCY_SCHEMA)
            result = json.loads(response)
            return result["score"]
        except (ValueError, AttributeError, json.JSONDecodeError):
            return 0.0
            
    async def _supports_concept(self, concept: Dict[str, Any], 
                         example: Dict[str, Any]) -> bool:
        """Check if an example supports the concept."""
        # Use LLM to validate
        validation = await self.llm.validate_concept(
            json.dumps(concept),
            example
        )
        return validation.confidence > self.confidence_threshold  # Consider it valid if confidence is high enough

class KnowledgeIntegrator:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.knowledge_base = {}
        
    async def integrate_concept(self, concept: Dict[str, Any], 
                         validation: ConceptValidation) -> Dict[str, Any]:
        """Integrate validated concept into knowledge base."""
        if not validation.is_valid:
            return {'success': False, 'reason': 'Invalid concept'}
            
        # Use LLM to analyze integration
        try:
            analysis = await self.llm.analyze_pattern({
                'concept': concept,
                'validation': validation,
                'knowledge_base': self.knowledge_base,
                'task': 'integration'
            })
            
            if not analysis:
                return {'success': False, 'reason': 'Analysis failed'}

            confidence = getattr(analysis, 'confidence', 0.5)
            
            integration = {
                'concept': concept,
                'validation': validation,
                'relationships': await self._find_knowledge_relationships(concept),
                'confidence': confidence
            }
            
            self.knowledge_base[concept['id']] = integration
            return {'success': True, 'integration': integration}
        
        except Exception as e:
            logger.error(f"Error integrating concept: {str(e)}")
            return {'success': False, 'reason': 'Integration failed'}
        
    async def _find_knowledge_relationships(self, 
                                    concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find relationships with existing knowledge."""
        relationships = []
        for existing in self.knowledge_base.values():
            # Use LLM to analyze relationship
            analysis = await self.llm.analyze_pattern({
                'concept1': concept,
                'concept2': existing['concept'],
                'task': 'relationship'
            })
            
            if analysis and hasattr(analysis, 'text'):
                try:
                    result = json.loads(analysis.text)
                    if result.get('relationship'):
                        relationships.append(result)
                except json.JSONDecodeError:
                    continue
                    
        return relationships

class ConceptFormationEngine:
    """System for learning high-level concepts from examples"""
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.concepts = []  # List to store learned concepts
        self.pattern_abstractor = PatternAbstractor(llm)
        self.concept_validator = ConceptValidator(llm)
        self.knowledge_integrator = KnowledgeIntegrator(llm)
        self.evolution_tracker = {}  # concept_id -> ConceptEvolution
        self.feedback_threshold = 0.7  # When to ask for feedback
        self.learning_history = []  # Track learning events
        
    async def learn_from_example(self, task_data: Dict[str, Any],
                          solution: Optional[Dict[str, Any]] = None,
                          feedback: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Learn concepts from example"""
        learned_items = []
        
        # Extract patterns from task
        patterns = await self._extract_patterns(task_data['input'])
        output_patterns = await self._extract_patterns(task_data['output'])
        
        # If we have feedback about borders, ensure we focus on them
        border_focus = ""
        if feedback and 'feedback' in feedback:
            if 'border' in feedback['feedback'].lower():
                border_focus = """
Special Instructions:
- Pay extra attention to border patterns and relationships
- Look for transformations that affect the border cells
- Consider relationships between border and non-border cells
- Focus on how borders change between input and output
"""
        
        # Analyze patterns for concepts
        prompt = f"""Analyze these patterns to discover concepts:

Input Patterns: {json.dumps(patterns, indent=2)}
Output Patterns: {json.dumps(output_patterns, indent=2)}

Previous Feedback: {json.dumps(feedback, indent=2)}
{border_focus}

Focus on:
1. Border patterns and relationships
2. Transformation patterns
3. Object patterns
4. Pattern relationships

Return concepts in JSON format:
{{
    "concepts": [
        {{
            "id": "concept_id",
            "name": "concept name",
            "description": "detailed description focusing on borders and transformations",
            "rules": ["list of rules"],
            "applicability": "when to use",
            "examples": ["example 1", "example 2"],
            "confidence": 0.0-1.0
        }}
    ]
}}"""

        max_retries = 3
        current_retry = 0
        
        while current_retry < max_retries:
            try:
                # Get LLM response with schema validation
                response = await self.llm.get_completion(prompt, schema=JSONValidator.CONCEPT_SCHEMA)
                is_valid, result, error = JSONValidator.validate_json(response, JSONValidator.CONCEPT_SCHEMA)
                
                if not is_valid:
                    # Generate fix prompt and retry
                    fix_prompt = JSONValidator.generate_fix_prompt(prompt, error, JSONValidator.CONCEPT_SCHEMA)
                    response = await self.llm.get_completion(fix_prompt, schema=JSONValidator.CONCEPT_SCHEMA)
                    is_valid, result, error = JSONValidator.validate_json(response, JSONValidator.CONCEPT_SCHEMA)
                    
                    if not is_valid:
                        current_retry += 1
                        continue
                
                # Process each concept
                for concept in result.get('concepts', []):
                    # Add border focus if needed
                    if border_focus and 'description' in concept:
                        if 'border' not in concept['description'].lower():
                            concept['description'] = f"Border-related concept: {concept['description']}"
                    
                    # Validate concept
                    validation = await self.concept_validator.validate_concept(concept, [{'input': task_data['input'], 'output': task_data['output']}])
                    
                    if validation.is_valid:
                        learned_items.append({
                            'type': 'concept',
                            'item': concept
                        })
                        
                        # Track evolution
                        await self._track_evolution(concept, validation, solution, feedback)
                
                return {'learned_items': learned_items}
                
            except Exception as e:
                print(f"Error learning concepts (attempt {current_retry + 1}): {str(e)}")
                current_retry += 1
        
        print("Failed to learn concepts after all retries")
        return {'learned_items': []}
        
    async def learn_from_examples(self, examples: List[Dict[str, Any]],
                            solution: Optional[Dict[str, Any]] = None,
                            feedback: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Learn unified concepts from multiple examples"""
        learned_items = []
        
        # Extract patterns from all examples
        all_input_patterns = []
        all_output_patterns = []
        for example in examples:
            input_patterns = await self._extract_patterns(example['input'])
            output_patterns = await self._extract_patterns(example['output'])
            all_input_patterns.append(input_patterns)
            all_output_patterns.append(output_patterns)
        
        # If we have feedback about borders, ensure we focus on them
        border_focus = ""
        if feedback and 'feedback' in feedback:
            if 'border' in feedback['feedback'].lower():
                border_focus = """
Special Instructions:
- Pay extra attention to border patterns and relationships
- Look for transformations that affect the border cells
- Consider relationships between border and non-border cells
- Focus on how borders change between input and output
"""
        
        # Analyze patterns across all examples to find unified concepts
        prompt = f"""Analyze these patterns to discover unified concepts that work for ALL examples:

Examples:
{json.dumps([{
    'input_patterns': inp,
    'output_patterns': out
} for inp, out in zip(all_input_patterns, all_output_patterns)], indent=2)}

Previous Feedback: {json.dumps(feedback, indent=2)}
{border_focus}

Instructions:
1. Find patterns that are consistent across ALL examples
2. Identify transformation rules that work for ALL examples
3. Note any important variations that the concepts must handle
4. Focus on generating concepts that can solve ALL examples

Focus on:
1. Border patterns and relationships
2. Transformation patterns
3. Object patterns
4. Pattern relationships

Return concepts in CONCEPT_SCHEMA format."""

        # Get concepts from LLM
        response = await self.llm.get_completion(prompt)
        
        try:
            concepts = json.loads(response)
            if not isinstance(concepts, dict) or 'concepts' not in concepts:
                raise ValueError("Invalid concept format")
                
            # Validate each concept works for all examples
            for concept in concepts['concepts']:
                validation = await self.concept_validator.validate_concept(concept, examples)
                if validation.is_valid:
                    # Only keep concepts that work for all examples
                    await self.knowledge_integrator.integrate_concept(concept, validation)
                    learned_items.append({
                        'type': 'concept',
                        'item': concept,
                        'validation': validation
                    })
                    
                    # Track evolution
                    await self._track_evolution(concept, validation, solution, feedback)
                    
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error processing concepts: {str(e)}")
            
        return {'learned_items': learned_items}

    async def apply_learned_concepts(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply learned concepts to solve a task"""
        # Find relevant concepts
        relevant_concepts = await self._find_relevant_concepts(task)
        
        # Generate solution steps
        steps = []
        explanations = []
        for concept in relevant_concepts:
            # Use LLM to analyze applicability
            prompt = f"""Analyze if this concept is applicable to the task:

Concept:
{json.dumps(concept, indent=2)}

Task:
{json.dumps(task, indent=2)}

Return a score between 0.0 and 1.0 indicating applicability."""
            
            try:
                response = await self.llm.get_completion(prompt)
                applicability_score = float(response.strip())
                
                if applicability_score > 0.7:
                    # Generate steps to apply the concept
                    prompt = f"""Generate steps to apply this concept to the task:

Concept:
{json.dumps(concept, indent=2)}

Task:
{json.dumps(task, indent=2)}

Return steps in JSON format:
{{
    "steps": [
        {{
            "step": "step description",
            "confidence": 0.0-1.0
        }}
    ]
}}"""
                    
                    response = await self.llm.get_completion(prompt)
                    try:
                        result = json.loads(response)
                        steps.extend(result['steps'])
                        explanations.extend([f"Applying concept {concept['name']}"] * len(result['steps']))
                    except (json.JSONDecodeError, KeyError):
                        pass
            except (ValueError, AttributeError):
                pass
                
        return {
            'steps': steps,
            'explanations': explanations,
            'concepts_used': relevant_concepts
        }
        
    async def incorporate_feedback(self, concept_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Incorporate feedback to improve concepts"""
        # Find the concept to update
        concept_to_update = None
        for concept in self.concepts:
            if concept['id'] == concept_id:
                concept_to_update = concept
                break
                
        if not concept_to_update:
            return
            
        # Update concept based on feedback
        prompt = f"""Update this concept based on the feedback:

Concept:
{json.dumps(concept_to_update, indent=2)}

Feedback:
{json.dumps(feedback, indent=2)}

Return updated concept in JSON format:
{{
    "updated_concept": {{
        "id": "{concept_id}",
        "name": "concept name",
        "description": "detailed description incorporating feedback",
        "rules": ["list of rules"],
        "applicability": "when this concept applies",
        "examples": ["example applications"]
    }}
}}"""

        try:
            response = await self.llm.get_completion(prompt)
            result = json.loads(response)
            updated_concept = result['updated_concept']
            
            # Update the concept in our list
            for i, concept in enumerate(self.concepts):
                if concept['id'] == concept_id:
                    self.concepts[i] = updated_concept
                    
                    # Track the evolution
                    if concept_id not in self.evolution_tracker:
                        self.evolution_tracker[concept_id] = ConceptEvolution(
                            concept_id=concept_id,
                            versions=[concept],  # Original version
                            performance_history=[],
                            feedback_history=[],
                            adaptation_history=[]
                        )
                    
                    # Add new version and feedback
                    self.evolution_tracker[concept_id].versions.append(updated_concept)
                    self.evolution_tracker[concept_id].feedback_history.append({
                        'timestamp': time.time(),
                        'feedback': feedback
                    })
                    break
                    
            # Add to learning history
            self.learning_history.append({
                'timestamp': time.time(),
                'event': 'feedback_incorporated',
                'concept_id': concept_id,
                'feedback': feedback,
                'result': updated_concept
            })
            
        except Exception as e:
            print(f"Error incorporating feedback: {str(e)}")
        
    async def _track_evolution(self, concept: Dict[str, Any],
                        validation: ConceptValidation,
                        solution: Optional[Dict[str, Any]] = None,
                        feedback: Optional[Dict[str, Any]] = None) -> None:
        """Track how a concept evolves"""
        if concept['id'] not in self.evolution_tracker:
            self.evolution_tracker[concept['id']] = ConceptEvolution(
                concept_id=concept['id'],
                versions=[concept],
                performance_history=[],
                feedback_history=[],
                adaptation_history=[]
            )
            
        evolution = self.evolution_tracker[concept['id']]
        
        # Record performance
        if solution:
            success_rate = validation.is_valid
            evolution.performance_history.append(success_rate)
            
        # Record feedback
        if feedback:
            evolution.feedback_history.append(feedback)
            
        # Check if adaptation is needed
        if len(evolution.performance_history) >= 5:
            recent_performance = evolution.performance_history[-5:]
            if np.mean(recent_performance) < 0.7:
                adaptation = await self._generate_adaptation(concept, {
                    'trigger': 'poor_performance',
                    'performance': recent_performance
                })
                if adaptation:
                    evolution.versions.append(adaptation)
                    evolution.adaptation_history.append({
                        'trigger': 'poor_performance',
                        'performance': recent_performance,
                        'adaptation': adaptation
                    })
                    
    async def _check_feedback_needed(self, concepts: List[Dict[str, Any]],
                             validations: List[ConceptValidation]) -> bool:
        """Check if human feedback is needed"""
        for concept, validation in zip(concepts, validations):
            # New concept with low confidence
            if len(self.evolution_tracker[concept['id']].versions) == 1 and \
               not validation.is_valid:
                return True
                
            # Concept with degrading performance
            if len(self.evolution_tracker[concept['id']].performance_history) >= 5:
                recent_performance = self.evolution_tracker[concept['id']].performance_history[-5:]
                if np.mean(recent_performance) < self.feedback_threshold:
                    return True
                    
        return False

    async def _generate_adaptation(self, concept: Dict[str, Any],
                           trigger: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate adaptation for a concept"""
        prompt = f"""This concept needs adaptation:

Concept:
{json.dumps(concept, indent=2)}

Trigger:
{json.dumps(trigger, indent=2)}

Suggest adaptations in JSON format:
{{
    "adapted_concept": {{
        "id": "concept_id",
        "name": "concept name",
        "description": "detailed description",
        "rules": ["list of rules that define this concept"],
        "applicability": "when this concept applies",
        "examples": ["example applications"],
        "adaptations": ["what was changed"]
    }}
}}"""
        
        response = await self.llm.get_completion(prompt)
        try:
            result = json.loads(response)
            return result['adapted_concept']
        except (json.JSONDecodeError, KeyError):
            return None

    async def _apply_feedback(self, concepts: List[Dict[str, Any]], 
                       feedback: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply feedback to concepts"""
        # Use LLM to apply feedback
        prompt = f"""Apply this feedback to the concepts:

Feedback:
{json.dumps(feedback, indent=2)}

Concepts:
{json.dumps(concepts, indent=2)}

Return updated concepts:
{{
    "updated_concepts": [
        {{
            "id": "concept_id",
            "name": "concept name",
            "description": "detailed description incorporating feedback",
            "rules": ["list of rules that define this concept"],
            "applicability": "when this concept applies",
            "examples": ["example applications"]
        }}
    ]
}}"""
        
        response = await self.llm.get_completion(prompt)
        try:
            return json.loads(response)['updated_concepts']
        except (json.JSONDecodeError, KeyError):
            return concepts

    async def _learn_from_solution(self, concepts: List[Dict[str, Any]], 
                            solution: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Learn from a solution"""
        # Use LLM to analyze solution
        prompt = f"""Analyze this solution and learn from it:

Solution:
{json.dumps(solution, indent=2)}

Concepts:
{json.dumps(concepts, indent=2)}

Return updated concepts:
{{
    "updated_concepts": [
        {{
            "id": "concept_id",
            "name": "concept name",
            "description": "detailed description",
            "rules": ["list of rules that define this concept"],
            "applicability": "when this concept applies",
            "examples": ["example applications"]
        }}
    ]
}}"""
        
        response = await self.llm.get_completion(prompt)
        try:
            updated_concepts = json.loads(response)['updated_concepts']
            for concept in concepts:
                updated_concept = next((uc for uc in updated_concepts if uc['id'] == concept['id']), None)
                if updated_concept:
                    concept.update(updated_concept)
        except (json.JSONDecodeError, KeyError):
            pass
            
    async def _find_relevant_concepts(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find concepts relevant to a task"""
        prompt = f"""Given this task, which concepts are most relevant?

Task:
{json.dumps(task, indent=2)}

Available Concepts:
{json.dumps(self.concepts, indent=2)}

Return relevant concept IDs and why they apply:
{{
    "relevant_concepts": [
        {{
            "id": "concept_id",
            "relevance_score": 0.0-1.0,
            "explanation": "why this concept is relevant"
        }}
    ]
}}"""
        
        response = await self.llm.get_completion(prompt)
        try:
            result = json.loads(response)
            concepts = []
            for item in result['relevant_concepts']:
                if item['relevance_score'] > 0.7:  # Relevance threshold
                    concept_id = item['id']
                    if concept_id in [concept['id'] for concept in self.concepts]:
                        concepts.append(next(concept for concept in self.concepts if concept['id'] == concept_id))
            return concepts
        except (json.JSONDecodeError, KeyError):
            return []
            
    async def _extract_patterns(self, grid: List[List[int]]) -> Dict[str, Any]:
        """Extract patterns from a grid"""
        try:
            patterns = {
                'dimensions': [len(grid), len(grid[0])],
                'unique_values': sorted(list(set(val for row in grid for val in row))),
                'row_patterns': [],
                'col_patterns': [],
                'region_patterns': []
            }

            # Row patterns
            for row in grid:
                patterns['row_patterns'].append({
                    'values': row,
                    'sum': sum(row),
                    'runs': await self._get_runs(row)
                })

            # Column patterns
            for j in range(len(grid[0])):
                col = [grid[i][j] for i in range(len(grid))]
                patterns['col_patterns'].append({
                    'values': col,
                    'sum': sum(col),
                    'runs': await self._get_runs(col)
                })

            # Region patterns (2x2)
            for i in range(len(grid)-1):
                for j in range(len(grid[0])-1):
                    region = [
                        grid[i][j], grid[i][j+1],
                        grid[i+1][j], grid[i+1][j+1]
                    ]
                    patterns['region_patterns'].append({
                        'position': [i, j],
                        'values': region,
                        'sum': sum(region)
                    })

            # Validate patterns
            is_valid, _, error = JSONValidator.validate_json(json.dumps(patterns), JSONValidator.PATTERN_SCHEMA)
            if not is_valid:
                print(f"Invalid pattern structure: {error}")
                return {
                    'dimensions': [0, 0],
                    'unique_values': [],
                    'row_patterns': [],
                    'col_patterns': [],
                    'region_patterns': []
                }

            return patterns

        except Exception as e:
            print(f"Error extracting patterns: {str(e)}")
            return {
                'dimensions': [0, 0],
                'unique_values': [],
                'row_patterns': [],
                'col_patterns': [],
                'region_patterns': []
            }

    async def _get_runs(self, values: List[int]) -> List[Dict[str, Any]]:
        """Get runs of same values"""
        if not values:
            return []

        runs = []
        current_val = values[0]
        current_len = 1

        for val in values[1:]:
            if val == current_val:
                current_len += 1
            else:
                runs.append({
                    'value': current_val,
                    'length': current_len
                })
                current_val = val
                current_len = 1

        runs.append({
            'value': current_val,
            'length': current_len
        })

        return runs

class ConceptEvolution:
    def __init__(self, concept_id: str, versions: List[Dict[str, Any]], 
                 performance_history: List[float], feedback_history: List[Dict[str, Any]], 
                 adaptation_history: List[Dict[str, Any]]):
        self.concept_id = concept_id
        self.versions = versions
        self.performance_history = performance_history
        self.feedback_history = feedback_history
        self.adaptation_history = adaptation_history
