from typing import Dict, List, Any
import numpy as np
from src.llm.llm_interface import LLMInterface

class ConceptExtractor:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.concepts = {}
        
    def extract_concepts(self, input_grid: np.ndarray, output_grid: np.ndarray) -> List[Dict[str, Any]]:
        """Extract concepts from input-output grid pair."""
        # Use LLM to extract concepts
        analysis = self.llm.analyze_pattern({
            'input_grid': input_grid.tolist(),
            'output_grid': output_grid.tolist(),
            'analysis_type': 'concept_extraction'
        })
        
        concepts = []
        for concept_desc in analysis.text.split('\n'):
            concept = self._create_concept(concept_desc, input_grid, output_grid)
            if concept:
                concepts.append(concept)
                self.concepts[concept['id']] = concept
                
        return concepts
        
    def _create_concept(self, description: str, input_grid: np.ndarray, 
                       output_grid: np.ndarray) -> Dict[str, Any]:
        """Create a concept from description and grids."""
        # Use LLM to validate and enrich concept
        validation = self.llm.validate_concept(
            description,
            {
                'input_grid': input_grid.tolist(),
                'output_grid': output_grid.tolist()
            }
        )
        
        if validation.confidence < 0.7:
            return None
            
        return {
            'id': f"concept_{len(self.concepts)}",
            'description': description,
            'confidence': validation.confidence,
            'examples': [{
                'input': input_grid.tolist(),
                'output': output_grid.tolist()
            }],
            'validation': validation.text.split('\n')
        }
        
    def get_concepts(self) -> List[Dict[str, Any]]:
        """Get all learned concepts."""
        return list(self.concepts.values())
        
    def update_from_feedback(self, feedback: str):
        """Update concepts based on feedback."""
        # Use LLM to analyze feedback
        analysis = self.llm.analyze_pattern({
            'feedback': feedback,
            'analysis_type': 'concept_feedback'
        })
        
        # Update concepts based on feedback
        for concept_id, update in zip(self.concepts.keys(), analysis.text.split('\n')):
            self.concepts[concept_id]['confidence'] = float(update)
            
    def update_from_execution(self, task: Dict[str, Any], 
                            solution: Dict[str, Any], 
                            success: bool):
        """Update concepts based on execution results."""
        if success:
            # Use LLM to extract new concepts from successful execution
            analysis = self.llm.analyze_pattern({
                'task': task,
                'solution': solution,
                'analysis_type': 'concept_extraction'
            })
            
            for concept_desc in analysis.text.split('\n'):
                concept = self._create_concept(
                    concept_desc,
                    np.array(task['train'][0]['input']),
                    np.array(task['train'][0]['output'])
                )
                if concept:
                    self.concepts[concept['id']] = concept
