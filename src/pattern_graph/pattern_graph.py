from typing import Dict, List, Tuple, Optional
import networkx as nx
import numpy as np
from dataclasses import dataclass

@dataclass
class Pattern:
    id: str
    features: np.ndarray
    transformations: List[str]
    frequency: int
    confidence: float

class PatternGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.patterns = {}
        
    def add_pattern(self, pattern: Pattern):
        """Add a new pattern to the graph."""
        self.patterns[pattern.id] = pattern
        self.graph.add_node(pattern.id, 
                          features=pattern.features,
                          transformations=pattern.transformations,
                          frequency=pattern.frequency,
                          confidence=pattern.confidence)
        
    def add_relationship(self, pattern1_id: str, pattern2_id: str, 
                        relationship_type: str, weight: float):
        """Add a relationship between two patterns."""
        if pattern1_id not in self.patterns or pattern2_id not in self.patterns:
            raise ValueError("Both patterns must exist in the graph")
            
        self.graph.add_edge(pattern1_id, pattern2_id,
                           type=relationship_type,
                           weight=weight)
        
    def find_related_patterns(self, pattern_id: str, 
                            relationship_type: Optional[str] = None) -> List[Tuple[str, float]]:
        """Find patterns related to the given pattern."""
        if pattern_id not in self.patterns:
            raise ValueError(f"Pattern {pattern_id} not found in graph")
            
        related = []
        for _, target, data in self.graph.edges(pattern_id, data=True):
            if relationship_type is None or data['type'] == relationship_type:
                related.append((target, data['weight']))
        return sorted(related, key=lambda x: x[1], reverse=True)
        
    def find_pattern_sequence(self, start_pattern: str, 
                            end_pattern: str) -> List[str]:
        """Find the most likely sequence of patterns between start and end patterns."""
        try:
            path = nx.shortest_path(self.graph, start_pattern, end_pattern,
                                  weight='weight', method='dijkstra')
            return path
        except nx.NetworkXNoPath:
            return []
            
    def update_pattern_confidence(self, pattern_id: str, new_confidence: float):
        """Update the confidence score of a pattern."""
        if pattern_id not in self.patterns:
            raise ValueError(f"Pattern {pattern_id} not found in graph")
            
        self.patterns[pattern_id].confidence = new_confidence
        self.graph.nodes[pattern_id]['confidence'] = new_confidence
        
    def get_most_frequent_patterns(self, n: int = 10) -> List[Pattern]:
        """Get the n most frequent patterns."""
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda x: x.frequency,
            reverse=True
        )
        return sorted_patterns[:n]
        
    def get_highest_confidence_patterns(self, n: int = 10) -> List[Pattern]:
        """Get the n patterns with highest confidence."""
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda x: x.confidence,
            reverse=True
        )
        return sorted_patterns[:n]
        
    def prune_low_confidence_patterns(self, threshold: float = 0.3):
        """Remove patterns with confidence below threshold."""
        patterns_to_remove = [
            p_id for p_id, p in self.patterns.items()
            if p.confidence < threshold
        ]
        for pattern_id in patterns_to_remove:
            self.graph.remove_node(pattern_id)
            del self.patterns[pattern_id]
