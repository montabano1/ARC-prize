from typing import Dict, List, Any
import numpy as np
from src.llm.llm_interface import LLMInterface

class StrategySelector:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.strategies = {}
        self.performance_history = {}
        
    def register_strategy(self, name: str, strategy_fn: callable, context_requirements: List[str]):
        """Register a new strategy with the selector."""
        self.strategies[name] = {
            'function': strategy_fn,
            'requirements': context_requirements,
            'performance': []
        }
        
    def select_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Select the best strategy based on context and historical performance."""
        # Use LLM to analyze context and select strategy
        analysis = self.llm.analyze_pattern({
            'context': context,
            'analysis_type': 'strategy_selection'
        })
        
        strategy_name = analysis.text.split('\n')[0]
        if strategy_name not in self.strategies:
            # Fallback to basic strategy selection
            valid_strategies = self._filter_valid_strategies(context)
            if not valid_strategies:
                raise ValueError("No valid strategies found for given context")
            strategy_name = self._select_best_strategy(valid_strategies)
            
        return {
            'name': strategy_name,
            'strategy': self.strategies[strategy_name],
            'confidence': analysis.confidence
        }
        
    def _filter_valid_strategies(self, context: Dict[str, Any]) -> List[str]:
        """Filter strategies based on context requirements."""
        valid_strategies = []
        for name, strategy in self.strategies.items():
            if all(req in context for req in strategy['requirements']):
                valid_strategies.append(name)
        return valid_strategies
        
    def _select_best_strategy(self, valid_strategies: List[str]) -> str:
        """Select the best strategy based on historical performance."""
        best_strategy = None
        best_performance = float('-inf')
        
        for strategy in valid_strategies:
            performance = self._calculate_strategy_performance(strategy)
            if performance > best_performance:
                best_performance = performance
                best_strategy = strategy
                
        return best_strategy or valid_strategies[0]
        
    def _calculate_strategy_performance(self, strategy: str) -> float:
        """Calculate performance score for a strategy."""
        if not self.strategies[strategy]['performance']:
            return 0.0
            
        performances = self.strategies[strategy]['performance']
        return sum(p['success'] for p in performances) / len(performances)
        
    def update_from_feedback(self, feedback: str):
        """Update strategy selection based on feedback."""
        # Use LLM to analyze feedback
        analysis = self.llm.analyze_pattern({
            'feedback': feedback,
            'analysis_type': 'strategy_feedback'
        })
        
        # Update performance history based on feedback
        for strategy, update in analysis.text.split('\n'):
            if strategy in self.strategies:
                self.strategies[strategy]['performance'].append({
                    'success': float(update),
                    'feedback': feedback
                })
