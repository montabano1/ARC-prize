from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from src.llm.llm_interface import LLMInterface
from src.meta_strategy.strategy_selector import StrategySelector
from src.concept_formation.concept_extractor import ConceptExtractor
from src.pattern_graph.pattern_graph import PatternGraph

@dataclass
class SolutionComponent:
    id: str
    type: str  # 'strategy', 'concept', 'pattern', 'transformation'
    content: Dict[str, Any]
    dependencies: List[str]
    confidence: float

@dataclass
class Solution:
    components: List[SolutionComponent]
    execution_order: List[str]
    validation_results: Dict[str, Any]
    performance_metrics: Dict[str, float]
    confidence: float

class ProgramComposer:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.solutions = {}
        
    def compose_solution(self, task: Dict[str, Any], 
                        components: List[SolutionComponent]) -> Solution:
        """Compose a complete solution from components."""
        # Use LLM to analyze component relationships
        analysis = self.llm.analyze_pattern({
            'task': task,
            'components': [comp.content for comp in components],
            'analysis_type': 'solution_composition'
        })
        
        # Determine execution order
        execution_order = self._determine_execution_order(components)
        
        # Create solution
        solution = Solution(
            components=components,
            execution_order=execution_order,
            validation_results={},
            performance_metrics={},
            confidence=analysis.confidence if hasattr(analysis, 'confidence') else 0.8
        )
        
        # Store solution with a generated ID if none exists
        task_id = task.get('id', f"task_{len(self.solutions)}")
        self.solutions[task_id] = solution
        return solution
        
    def _determine_execution_order(self, 
                                 components: List[SolutionComponent]) -> List[str]:
        """Determine optimal execution order of components."""
        # Sort components by type priority
        type_priority = {
            'border transformation': 1,  # First set all border points to 1
            'central point': 2,         # Then handle the center point
            'symmetry': 3               # Finally ensure symmetry
        }
        
        def get_priority(component):
            desc = component.content.get('description', '').lower()
            for key, priority in type_priority.items():
                if key in desc:
                    return priority
            return 99  # Default low priority
            
        sorted_components = sorted(components, key=get_priority)
        return [comp.id for comp in sorted_components]
        
    def _validate_solution(self, solution: Solution) -> Dict[str, Any]:
        """Validate the composed solution."""
        # Use LLM to validate solution
        validation = self.llm.validate_concept(
            "Validate this solution composition",
            {'solution': solution}
        )
        
        return {
            'is_valid': validation.confidence > 0.8,
            'confidence': validation.confidence,
            'issues': self._identify_issues(solution)
        }
        
    def _identify_issues(self, solution: Solution) -> List[str]:
        """Identify potential issues in the solution."""
        # Use LLM to identify issues
        analysis = self.llm.analyze_pattern({
            'solution': solution,
            'task': 'identify_issues'
        })
        
        return analysis.text.split('\n') if analysis.confidence > 0.6 else []

class StrategyExecutor:
    def __init__(self, strategy_selector: StrategySelector, llm: LLMInterface):
        self.strategy_selector = strategy_selector
        self.llm = llm
        self.execution_history = {}
        
    def execute_solution(self, solution: Solution, 
                        task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a composed solution."""
        results = {}
        current_grid = np.array(task['train'][0]['input'])
        
        for component_id in solution.execution_order:
            component = next(c for c in solution.components if c.id == component_id)
            
            # Execute component based on type
            if component.type == 'concept':
                result = self._apply_concept(component, current_grid)
                if result is not None:
                    current_grid = result
                    results[component_id] = {'grid': result}
                    print(f"\nAfter applying: {component.content['description']}")
                    self._print_grid(result)
                    
        results['final_grid'] = current_grid
        aggregated = self._aggregate_results(results)
        
        # Update solution with results
        solution.validation_results = aggregated
        
        return aggregated
        
    def _print_grid(self, grid: np.ndarray) -> None:
        """Helper to print a grid."""
        for row in grid:
            print(" ".join(str(x) for x in row))
        
    def _apply_concept(self, component: SolutionComponent, 
                      grid: np.ndarray) -> np.ndarray:
        """Apply a concept component to transform the grid."""
        desc = component.content.get('description', '').lower()
        result = grid.copy()
        center = tuple(d // 2 for d in grid.shape)
        
        if 'border transformation' in desc:
            # Set all non-center points to 1
            rows, cols = grid.shape
            for i in range(rows):
                for j in range(cols):
                    if (i, j) != center:
                        result[i, j] = 1
            return result
            
        elif 'central point' in desc:
            # Set center point to 0
            result[center] = 0
            return result
            
        elif 'symmetry' in desc or 'neighborhood' in desc:
            # Ensure all non-center points are 1
            rows, cols = grid.shape
            for i in range(rows):
                for j in range(cols):
                    if (i, j) != center:
                        result[i, j] = 1
            return result
            
        return None
        
    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from all components."""
        return {
            'success': True,
            'final_grid': results.get('final_grid', None),
            'steps': results
        }
        
    def _execute_strategy(self, component: SolutionComponent, 
                         task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a strategy component."""
        strategy = self.strategy_selector.get_strategy(component.content['strategy_id'])
        return strategy.execute(task)
        
    def _apply_pattern(self, component: SolutionComponent, 
                      task: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a pattern component."""
        # Implementation would depend on pattern type
        return {}
        
    def _apply_transformation(self, component: SolutionComponent, 
                            task: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a transformation component."""
        # Implementation would depend on transformation type
        return {}
        
    def _calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics from results."""
        return {
            'success_rate': sum(1 for r in results.values() 
                              if r.get('success', False)) / len(results),
            'average_confidence': np.mean([r.get('confidence', 0) 
                                         for r in results.values()])
        }

class ConceptApplier:
    def __init__(self, concept_extractor: ConceptExtractor, llm: LLMInterface):
        self.concept_extractor = concept_extractor
        self.llm = llm
        self.application_history = {}
        
    def apply_concepts(self, task: Dict[str, Any], 
                      concepts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply concepts to solve a task."""
        results = {}
        for concept in concepts:
            # Use LLM to analyze concept applicability
            analysis = self.llm.analyze_pattern({
                'task': task,
                'concept': concept
            })
            
            if analysis.confidence > 0.7:
                results[concept['id']] = self._apply_concept(concept, task)
                
        return self._aggregate_results(results)
        
    def _apply_concept(self, concept: Dict[str, Any], 
                      task: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a single concept to the task."""
        # Implementation would depend on concept type
        return {}
        
    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from concept application."""
        return {
            'success': any(r.get('success', False) for r in results.values()),
            'results': results,
            'confidence': np.mean([r.get('confidence', 0) for r in results.values()])
        }

class SolutionValidator:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.validation_history = {}
        
    def validate_solution(self, solution: Solution, 
                         task: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a complete solution."""
        # Get final grid from solution execution
        final_grid = solution.validation_results.get('final_grid')
        if final_grid is None:
            return {
                'success': False,
                'issues': ['No final grid produced'],
                'suggestions': ['Check that all transformations produce valid output']
            }
            
        # Compare with expected output
        expected_grid = np.array(task['train'][0]['output'])
        if np.array_equal(final_grid, expected_grid):
            return {
                'success': True,
                'performance_metrics': {
                    'accuracy': 1.0,
                    'steps': len(solution.execution_order)
                }
            }
            
        # If grids don't match, analyze differences
        issues = []
        suggestions = []
        
        # Check grid shape
        if final_grid.shape != expected_grid.shape:
            issues.append(f"Output grid shape {final_grid.shape} doesn't match expected shape {expected_grid.shape}")
            suggestions.append("Ensure transformations preserve grid dimensions")
            
        # Check value differences
        if final_grid.shape == expected_grid.shape:
            diff_count = np.sum(final_grid != expected_grid)
            if diff_count > 0:
                issues.append(f"Output grid differs from expected in {diff_count} positions")
                suggestions.append("Review transformation order and logic")
                
        # Get LLM analysis of differences
        analysis = self.llm.analyze_pattern({
            'input_grid': final_grid.tolist(),
            'output_grid': expected_grid.tolist(),
            'analysis_type': 'validation'
        })
        
        if hasattr(analysis, 'text') and analysis.text:
            suggestions.extend(analysis.text.split('\n'))
            
        return {
            'success': False,
            'issues': issues,
            'suggestions': suggestions,
            'performance_metrics': {
                'accuracy': 1 - (diff_count / expected_grid.size if 'diff_count' in locals() else 0),
                'steps': len(solution.execution_order)
            }
        }
        
    def _identify_issues(self, solution: Solution, 
                        task: Dict[str, Any]) -> List[str]:
        """Identify potential issues in the solution."""
        # Use LLM to identify issues
        analysis = self.llm.analyze_pattern({
            'solution': solution,
            'task': task,
            'analysis_type': 'issues'
        })
        
        return analysis.text.split('\n') if analysis.confidence > 0.6 else []
        
    def _generate_suggestions(self, solution: Solution, 
                            task: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving the solution."""
        # Use LLM to generate suggestions
        analysis = self.llm.analyze_pattern({
            'solution': solution,
            'task': task,
            'analysis_type': 'suggestions'
        })
        
        return analysis.text.split('\n') if analysis.confidence > 0.6 else []

class PerformanceOptimizer:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.optimization_history = {}
        
    def optimize_solution(self, solution: Solution, 
                         performance_metrics: Dict[str, float]) -> Solution:
        """Optimize a solution based on performance metrics."""
        # Use LLM to suggest optimizations
        optimization = self.llm.analyze_pattern({
            'solution': solution,
            'metrics': performance_metrics,
            'analysis_type': 'optimization'
        })
        
        if optimization.confidence > 0.7:
            optimized_solution = self._apply_optimizations(solution, optimization.text)
            
            # Validate optimized solution
            validation = self._validate_optimization(optimized_solution, solution)
            if validation['is_valid']:
                return optimized_solution
                
        return solution
        
    def _apply_optimizations(self, solution: Solution, 
                           optimizations: str) -> Solution:
        """Apply suggested optimizations to the solution."""
        # Implementation would depend on optimization type
        return solution
        
    def _validate_optimization(self, optimized: Solution, 
                             original: Solution) -> Dict[str, Any]:
        """Validate that optimization improved the solution."""
        # Use LLM to validate optimization
        validation = self.llm.validate_concept(
            "Validate this optimization",
            {'optimized': optimized, 'original': original}
        )
        
        return {
            'is_valid': validation.confidence > 0.8,
            'confidence': validation.confidence,
            'improvements': self._identify_improvements(optimized, original)
        }
        
    def _identify_improvements(self, optimized: Solution, 
                             original: Solution) -> List[str]:
        """Identify improvements made by optimization."""
        # Use LLM to identify improvements
        analysis = self.llm.analyze_pattern({
            'optimized': optimized,
            'original': original,
            'analysis_type': 'improvements'
        })
        
        return analysis.text.split('\n') if analysis.confidence > 0.6 else []
