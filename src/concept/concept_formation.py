from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.llm.llm_interface import LLMInterface

@dataclass
class Concept:
    """A learned concept that can be applied to solve tasks"""
    name: str
    description: str
    applicability_conditions: List[str]
    implementation_hints: List[str]
    examples: List[Dict[str, Any]]
    dsl_template: Optional[List[Dict[str, Any]]] = None  # DSL program template
    usage_count: int = 0
    success_rate: float = 0.0

class ConceptFormation:
    """System for learning and evolving concepts"""
    def __init__(self, llm: LLMInterface):
        """Initialize concept formation."""
        self.llm = llm
        self.concepts: Dict[str, Concept] = {}
        self.concept_relationships: Dict[str, List[str]] = {}
        
    async def extract_concepts(self, task_features: 'TaskFeatures') -> List[Concept]:
        """Extract concepts from task features."""
        try:
            # Use LLM to identify potential concepts
            prompt = f"""
            Analyze these task features and identify key concepts:
            
            Grid Size: {task_features.grid_size}
            Pattern Complexity: {task_features.pattern_complexity}
            Identified Patterns: {task_features.identified_patterns}
            
            Extract fundamental concepts that explain:
            1. The transformation rules
            2. Pattern relationships
            3. Spatial arrangements
            4. Color/value patterns
            
            Return a JSON array of concepts, each with:
            - id: string
            - name: string
            - description: string
            - confidence: float (0-1)
            """
            
            response = await self.llm.get_completion(prompt)
            
            try:
                import json
                concepts_data = json.loads(response)
                
                # Convert to Concept objects
                concepts = []
                for data in concepts_data:
                    concept = Concept(
                        name=data.get('name', ''),
                        description=data.get('description', ''),
                        applicability_conditions=[],
                        implementation_hints=[],
                        examples=[],  # Will be populated during validation
                        dsl_template=None
                    )
                    
                    # Validate concept
                    validation = await self._validate_concept(concept, task_features)
                    concept.validation_results = validation
                    
                    if validation.get('is_valid', False):
                        concepts.append(concept)
                        self.concepts[concept.name] = concept
                        
                return concepts
                
            except json.JSONDecodeError:
                print("Failed to parse concepts from LLM response")
                return []
                
        except Exception as e:
            print(f"Error extracting concepts: {str(e)}")
            return []
        
    def _extract_unified_strategy(self, patterns: Dict[str, List[Dict[str, Any]]]) -> Optional[Concept]:
        """Extract the unified strategy as a high-level concept"""
        # Look for unified strategy in abstract patterns
        for pattern in patterns.get('abstract', []):
            if pattern.get('is_unified_strategy'):
                desc = pattern.get('description', '')
                if not desc:
                    continue
                    
                # Create DSL template based on strategy description
                dsl_template = self._create_dsl_template(desc)
                
                return Concept(
                    name=f"strategy_{len(self.concepts)}",
                    description=desc,
                    applicability_conditions=self._extract_conditions(desc),
                    implementation_hints=self._extract_implementation_hints(desc),
                    examples=[],  # Will be filled with task examples
                    dsl_template=dsl_template
                )
        return None
        
    def _create_dsl_template(self, strategy_desc: str) -> List[Dict[str, Any]]:
        """Create a DSL template from a strategy description"""
        template = []
        desc = strategy_desc.lower()
        
        # Extract key operations from description
        if 'invert' in desc:
            if 'border' in desc:
                template.append({
                    'primitive': 'get_border',
                    'params': {}
                })
                template.append({
                    'primitive': 'invert_region',
                    'params': {'region': 'border'}
                })
            else:
                template.append({
                    'primitive': 'invert',
                    'params': {}
                })
                
        return template
        
    def _extract_conditions(self, desc: str) -> List[str]:
        """Extract applicability conditions from description"""
        conditions = []
        desc = desc.lower()
        
        # Extract conditions based on key phrases
        if 'border' in desc:
            conditions.append("Input has identifiable border region")
        if 'invert' in desc:
            conditions.append("Operation involves color/value inversion")
            
        return conditions
        
    def _extract_implementation_hints(self, desc: str) -> List[str]:
        """Extract implementation hints from description"""
        hints = []
        desc = desc.lower()
        
        # Extract hints based on key phrases
        if 'border' in desc:
            hints.append("Need to identify and isolate border region")
        if 'invert' in desc:
            hints.append("Need color/value inversion capability")
            
        return hints
        
    def _extract_category_concepts(self, category: str, 
                                 patterns: List[Dict[str, Any]]) -> List[Concept]:
        """Extract concepts from patterns in a specific category"""
        concepts = []
        
        for pattern in patterns:
            # Skip unified strategy as it's handled separately
            if pattern.get('is_unified_strategy'):
                continue
                
            desc = pattern.get('description', '')
            if not desc:
                continue
                
            # Create concept based on pattern
            concept = Concept(
                name=f"{category}_{len(self.concepts)}",
                description=desc,
                applicability_conditions=self._extract_conditions(desc),
                implementation_hints=self._extract_implementation_hints(desc),
                examples=[],
                dsl_template=self._create_dsl_template(desc)
            )
            
            concepts.append(concept)
            
        return concepts
        
    def _update_concept_relationships(self, concepts: List[Concept]):
        """Update relationships between concepts"""
        for concept in concepts:
            related = []
            for other in concepts:
                if concept != other:
                    # Check for related terms in descriptions
                    if any(term in other.description.lower() 
                          for term in concept.description.lower().split()):
                        related.append(other.name)
            self.concept_relationships[concept.name] = related

    async def _validate_concept(self, concept: Concept, task_features: 'TaskFeatures') -> Dict[str, Any]:
        # TO DO: implement concept validation logic
        return {'is_valid': True}
