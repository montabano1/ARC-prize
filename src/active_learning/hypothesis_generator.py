from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from src.concept_formation.concept_extractor import Concept
from src.pattern_graph.pattern_graph import Pattern

@dataclass
class Hypothesis:
    id: str
    concepts: List[Concept]
    patterns: List[Pattern]
    transformation_sequence: List[str]
    confidence: float
    validation_score: float

class HypothesisGenerator:
    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.hypotheses = []
        self.validated_hypotheses = []
        
    def generate_hypotheses(self, input_grids: List[np.ndarray],
                          output_grids: List[np.ndarray],
                          concepts: List[Concept],
                          patterns: List[Pattern]) -> List[Hypothesis]:
        """Generate hypotheses based on observed input-output pairs."""
        new_hypotheses = []
        
        # Generate hypotheses combining concepts and patterns
        for i in range(len(input_grids)):
            input_grid = input_grids[i]
            output_grid = output_grids[i]
            
            # Find relevant concepts
            relevant_concepts = self._find_relevant_concepts(input_grid, concepts)
            
            # Find relevant patterns
            relevant_patterns = self._find_relevant_patterns(input_grid, patterns)
            
            # Generate transformation sequences
            transformations = self._generate_transformations(
                input_grid, output_grid, relevant_concepts, relevant_patterns)
                
            # Create hypotheses
            for trans_seq in transformations:
                hypothesis = Hypothesis(
                    id=f"hyp_{len(self.hypotheses)}",
                    concepts=relevant_concepts,
                    patterns=relevant_patterns,
                    transformation_sequence=trans_seq,
                    confidence=self._calculate_initial_confidence(
                        relevant_concepts, relevant_patterns),
                    validation_score=0.0
                )
                new_hypotheses.append(hypothesis)
                
        # Filter hypotheses based on confidence
        filtered_hypotheses = [h for h in new_hypotheses 
                             if h.confidence >= self.confidence_threshold]
        
        self.hypotheses.extend(filtered_hypotheses)
        return filtered_hypotheses
        
    def validate_hypothesis(self, hypothesis: Hypothesis,
                          validation_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Validate a hypothesis against new input-output pairs."""
        correct_predictions = 0
        total_pairs = len(validation_pairs)
        
        for input_grid, expected_output in validation_pairs:
            predicted_output = self._apply_hypothesis(input_grid, hypothesis)
            if predicted_output is not None and np.array_equal(predicted_output, expected_output):
                correct_predictions += 1
                
        validation_score = correct_predictions / total_pairs
        hypothesis.validation_score = validation_score
        
        if validation_score >= self.confidence_threshold:
            self.validated_hypotheses.append(hypothesis)
            
        return validation_score
        
    def _find_relevant_concepts(self, grid: np.ndarray,
                              concepts: List[Concept]) -> List[Concept]:
        """Find concepts that are relevant to the given grid."""
        return [concept for concept in concepts
                if self._matches_concept(grid, concept)]
                
    def _find_relevant_patterns(self, grid: np.ndarray,
                              patterns: List[Pattern]) -> List[Pattern]:
        """Find patterns that are relevant to the given grid."""
        return [pattern for pattern in patterns
                if self._matches_pattern(grid, pattern)]
                
    def _generate_transformations(self, input_grid: np.ndarray,
                                output_grid: np.ndarray,
                                concepts: List[Concept],
                                patterns: List[Pattern]) -> List[List[str]]:
        """Generate possible transformation sequences."""
        transformations = []
        
        # Combine concept and pattern transformations
        all_transforms = []
        for concept in concepts:
            all_transforms.extend(concept.transformations)
        for pattern in patterns:
            all_transforms.extend(pattern.transformations)
            
        # Generate sequences of different lengths
        for length in range(1, 4):  # Limit to sequences of length 1-3
            sequences = self._generate_sequences(all_transforms, length)
            transformations.extend(sequences)
            
        return transformations
        
    def _generate_sequences(self, transforms: List[str], length: int) -> List[List[str]]:
        """Generate all possible sequences of given length."""
        if length == 1:
            return [[t] for t in transforms]
            
        sequences = []
        sub_sequences = self._generate_sequences(transforms, length - 1)
        for t in transforms:
            for seq in sub_sequences:
                sequences.append([t] + seq)
                
        return sequences
        
    def _calculate_initial_confidence(self, concepts: List[Concept],
                                   patterns: List[Pattern]) -> float:
        """Calculate initial confidence score for a hypothesis."""
        if not concepts and not patterns:
            return 0.0
            
        concept_confidence = np.mean([c.confidence for c in concepts]) if concepts else 0.0
        pattern_confidence = np.mean([p.confidence for p in patterns]) if patterns else 0.0
        
        # Weight concept confidence slightly higher than pattern confidence
        return 0.6 * concept_confidence + 0.4 * pattern_confidence
        
    def _matches_concept(self, grid: np.ndarray, concept: Concept) -> bool:
        """Check if a grid matches a concept."""
        # Implementation would depend on concept representation
        return True
        
    def _matches_pattern(self, grid: np.ndarray, pattern: Pattern) -> bool:
        """Check if a grid matches a pattern."""
        # Implementation would depend on pattern representation
        return True
        
    def _apply_hypothesis(self, input_grid: np.ndarray,
                         hypothesis: Hypothesis) -> Optional[np.ndarray]:
        """Apply a hypothesis to an input grid."""
        try:
            current_grid = input_grid.copy()
            for transform in hypothesis.transformation_sequence:
                # Apply transformation
                # Implementation would depend on transformation representation
                pass
            return current_grid
        except Exception:
            return None
