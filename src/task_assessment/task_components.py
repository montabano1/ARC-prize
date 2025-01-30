from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from src.llm.llm_interface import LLMInterface

@dataclass
class TaskFeatures:
    object_count: int
    transformation_types: List[str]
    step_complexity: float
    concept_difficulty: float
    extracted_features: Dict[str, Any]
    relationships: List[Dict[str, Any]]

class ObjectCounter:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
    def count_objects(self, grid: np.ndarray) -> Dict[str, Any]:
        """Count and classify objects in the grid."""
        unique_values = np.unique(grid)
        connected_components = self._find_connected_components(grid)
        
        # Use LLM to analyze objects
        analysis = self.llm.analyze_pattern({
            'grid': grid.tolist(),
            'components': connected_components,
            'analysis_type': 'object_counting'
        })
        
        return {
            'unique_values': len(unique_values),
            'connected_components': len(connected_components),
            'object_types': analysis.text.split('\n'),
            'confidence': analysis.confidence
        }
        
    def _find_connected_components(self, grid: np.ndarray) -> List[np.ndarray]:
        """Find connected components in the grid."""
        components = []
        visited = np.zeros_like(grid, dtype=bool)
        
        def dfs(i: int, j: int, value: int) -> np.ndarray:
            if (i < 0 or i >= grid.shape[0] or 
                j < 0 or j >= grid.shape[1] or 
                visited[i, j] or 
                grid[i, j] != value):
                return np.array([])
                
            visited[i, j] = True
            component = [(i, j)]
            
            # Check all 4 directions
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                component.extend(dfs(i + di, j + dj, value))
                
            return np.array(component)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not visited[i, j]:
                    component = dfs(i, j, grid[i, j])
                    if len(component) > 0:
                        components.append(component)
                        
        return components

class TransformationAnalyzer:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
    def analyze_transformation(self, input_grid: np.ndarray, 
                             output_grid: np.ndarray) -> Dict[str, Any]:
        """Analyze transformation between input and output grids."""
        # Use LLM to analyze transformation
        analysis = self.llm.analyze_pattern({
            'input_grid': input_grid.tolist(),
            'output_grid': output_grid.tolist(),
            'analysis_type': 'transformation'
        })
        
        return {
            'transformation_type': analysis.text.split('\n')[0],
            'steps': analysis.text.split('\n')[1:],
            'complexity': self._calculate_transformation_complexity(input_grid, output_grid),
            'confidence': analysis.confidence
        }
        
    def _calculate_transformation_complexity(self, input_grid: np.ndarray, 
                                          output_grid: np.ndarray) -> float:
        """Calculate complexity of transformation."""
        differences = np.sum(input_grid != output_grid)
        max_size = input_grid.size
        return differences / max_size

class StepComplexityEstimator:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
    def estimate_complexity(self, steps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate complexity of solution steps."""
        # Use LLM to analyze step complexity
        analysis = self.llm.analyze_pattern({
            'steps': steps,
            'analysis_type': 'step_complexity'
        })
        
        return {
            'overall_complexity': float(analysis.text.split('\n')[0]),
            'step_scores': self._calculate_step_scores(steps),
            'bottlenecks': analysis.text.split('\n')[1:],
            'confidence': analysis.confidence
        }
        
    def _calculate_step_scores(self, steps: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate complexity scores for individual steps."""
        scores = {}
        for i, step in enumerate(steps):
            # Use LLM to score step
            analysis = self.llm.analyze_pattern({
                'step': step,
                'analysis_type': 'step_score'
            })
            scores[f'step_{i}'] = float(analysis.text)
        return scores

class ConceptDifficultyEvaluator:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
    def evaluate_difficulty(self, concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate difficulty of required concepts."""
        # Use LLM to evaluate concept difficulty
        analysis = self.llm.analyze_pattern({
            'concepts': concepts,
            'analysis_type': 'concept_difficulty'
        })
        
        return {
            'overall_difficulty': float(analysis.text.split('\n')[0]),
            'concept_scores': self._calculate_concept_scores(concepts),
            'prerequisites': analysis.text.split('\n')[1:],
            'confidence': analysis.confidence
        }
        
    def _calculate_concept_scores(self, concepts: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate difficulty scores for individual concepts."""
        scores = {}
        for concept in concepts:
            # Use LLM to score concept
            analysis = self.llm.analyze_pattern({
                'concept': concept,
                'analysis_type': 'concept_score'
            })
            scores[concept['id']] = float(analysis.text)
        return scores

class FeatureExtractor:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
    def extract_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract relevant features from the grid."""
        # Basic statistical features
        statistical_features = {
            'mean': float(np.mean(grid)),
            'std': float(np.std(grid)),
            'unique_values': len(np.unique(grid))
        }
        
        # Use LLM to extract high-level features
        analysis = self.llm.analyze_pattern({
            'grid': grid.tolist(),
            'analysis_type': 'feature_extraction'
        })
        
        return {
            'statistical': statistical_features,
            'structural': self._extract_structural_features(grid),
            'semantic': analysis.text.split('\n'),
            'confidence': analysis.confidence
        }
        
    def _extract_structural_features(self, grid: np.ndarray) -> Dict[str, Any]:
        """Extract structural features from the grid."""
        return {
            'symmetry': self._calculate_symmetry(grid),
            'density': np.count_nonzero(grid) / grid.size,
            'edge_ratio': self._calculate_edge_ratio(grid)
        }
        
    def _calculate_symmetry(self, grid: np.ndarray) -> Dict[str, float]:
        """Calculate various symmetry scores."""
        h_sym = np.all(grid == np.fliplr(grid))
        v_sym = np.all(grid == np.flipud(grid))
        d_sym = np.all(grid == grid.T)
        
        return {
            'horizontal': float(h_sym),
            'vertical': float(v_sym),
            'diagonal': float(d_sym)
        }
        
    def _calculate_edge_ratio(self, grid: np.ndarray) -> float:
        """Calculate ratio of edge elements to total elements."""
        edges = np.concatenate([
            grid[0, :], grid[-1, :],
            grid[1:-1, 0], grid[1:-1, -1]
        ])
        return np.count_nonzero(edges) / len(edges)

class RelationshipAnalyzer:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
    def analyze_relationships(self, components: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze relationships between components."""
        # Use LLM to analyze relationships
        analysis = self.llm.analyze_pattern({
            'components': components,
            'analysis_type': 'relationships'
        })
        
        return {
            'direct': self._find_direct_relationships(components),
            'indirect': self._find_indirect_relationships(components),
            'hierarchical': analysis.text.split('\n'),
            'confidence': analysis.confidence
        }
        
    def _find_direct_relationships(self, 
                                 components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find direct relationships between components."""
        relationships = []
        for i, c1 in enumerate(components):
            for c2 in components[i+1:]:
                # Use LLM to analyze relationship
                analysis = self.llm.analyze_pattern({
                    'component1': c1,
                    'component2': c2,
                    'analysis_type': 'direct_relationship'
                })
                
                if analysis.confidence > 0.7:
                    relationships.append({
                        'components': [c1['id'], c2['id']],
                        'type': analysis.text,
                        'confidence': analysis.confidence
                    })
                    
        return relationships
        
    def _find_indirect_relationships(self, 
                                   components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find indirect relationships between components."""
        # Use LLM to find indirect relationships
        analysis = self.llm.analyze_pattern({
            'components': components,
            'analysis_type': 'indirect_relationships'
        })
        
        return [
            {
                'components': rel.split(':')[0].split(','),
                'type': rel.split(':')[1],
                'confidence': analysis.confidence
            }
            for rel in analysis.text.split('\n')
        ]
