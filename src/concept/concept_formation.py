from typing import List, Dict, Any, Optional
from dataclasses import dataclass

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
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.concept_relationships: Dict[str, List[str]] = {}
        
    def extract_concepts(self, task_features: 'TaskFeatures') -> List[Concept]:
        """Extract concepts from identified patterns"""
        concepts = []
        patterns = task_features.identified_patterns
        
        # First look for unified strategy
        unified_strategy = self._extract_unified_strategy(patterns)
        if unified_strategy:
            concepts.append(unified_strategy)
        
        # Then extract supporting concepts from each category
        for category, category_patterns in patterns.items():
            category_concepts = self._extract_category_concepts(category, category_patterns)
            concepts.extend(category_concepts)
            
        # Look for relationships between concepts
        self._update_concept_relationships(concepts)
        
        # Store new concepts
        for concept in concepts:
            if concept.name not in self.concepts:
                self.concepts[concept.name] = concept
            else:
                # Update existing concept with new examples
                existing = self.concepts[concept.name]
                existing.examples.extend(concept.examples)
                if concept.dsl_template:
                    existing.dsl_template = concept.dsl_template
                
        return concepts
        
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
