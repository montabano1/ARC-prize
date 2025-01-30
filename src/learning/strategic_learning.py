from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from src.llm.llm_interface import LLMInterface
from src.meta_strategy.strategy_selector import StrategySelector

@dataclass
class StrategyPerformance:
    strategy_id: str
    success_rate: float
    execution_time: float
    resource_usage: Dict[str, float]
    context_performance: Dict[str, float]

class StrategicLearner:
    def __init__(self, strategy_selector: StrategySelector, llm: LLMInterface):
        self.strategy_selector = strategy_selector
        self.llm = llm
        self.performance_history = {}
        self.learning_rate = 0.1
        
    def learn_from_execution(self, strategy: str, 
                           result: Dict[str, Any], 
                           context: Dict[str, Any]) -> None:
        """Learn from strategy execution results."""
        # Update performance history
        if strategy not in self.performance_history:
            self.performance_history[strategy] = []
            
        performance = StrategyPerformance(
            strategy_id=strategy,
            success_rate=float(result['success']),
            execution_time=result['execution_time'],
            resource_usage=result['resource_usage'],
            context_performance=self._evaluate_context_performance(strategy, context)
        )
        
        self.performance_history[strategy].append(performance)
        
        # Update strategy weights
        self._update_strategy_weights(strategy, performance)
        
        # Get LLM insights
        self._analyze_performance(strategy, performance)
        
    def _evaluate_context_performance(self, strategy: str, 
                                    context: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate strategy performance in specific context."""
        # Use LLM to analyze context performance
        analysis = self.llm.analyze_pattern({
            'strategy': strategy,
            'context': context
        })
        
        return {
            'relevance': analysis.confidence,
            'effectiveness': self._calculate_effectiveness(strategy, context)
        }
        
    def _update_strategy_weights(self, strategy: str, 
                               performance: StrategyPerformance) -> None:
        """Update strategy selection weights based on performance."""
        current_weight = self.strategy_selector.get_strategy_weight(strategy)
        performance_score = self._calculate_performance_score(performance)
        
        # Update weight using learning rate
        new_weight = current_weight + self.learning_rate * (performance_score - current_weight)
        self.strategy_selector.update_strategy_weight(strategy, new_weight)
        
    def _calculate_performance_score(self, 
                                  performance: StrategyPerformance) -> float:
        """Calculate overall performance score."""
        return (0.4 * performance.success_rate +
                0.3 * (1.0 - performance.execution_time / 100.0) +
                0.3 * np.mean(list(performance.context_performance.values())))
        
    def _analyze_performance(self, strategy: str, 
                           performance: StrategyPerformance) -> Dict[str, Any]:
        """Get LLM insights on strategy performance."""
        analysis = self.llm.analyze_pattern({
            'strategy': strategy,
            'performance': performance,
            'analysis_type': 'performance'
        })
        
        return {
            'insights': analysis.text,
            'confidence': analysis.confidence
        }

class PerformanceAnalyzer:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.analysis_history = {}
        
    def analyze_performance(self, strategy: str, 
                          performances: List[StrategyPerformance]) -> Dict[str, Any]:
        """Analyze strategy performance over time."""
        trends = self._analyze_trends(performances)
        context_impact = self._analyze_context_impact(performances)
        
        # Get LLM insights
        analysis = self.llm.analyze_pattern({
            'strategy': strategy,
            'trends': trends,
            'context_impact': context_impact
        })
        
        result = {
            'trends': trends,
            'context_impact': context_impact,
            'insights': analysis.text,
            'confidence': analysis.confidence
        }
        
        self.analysis_history[strategy] = result
        return result
        
    def _analyze_trends(self, 
                       performances: List[StrategyPerformance]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if not performances:
            return {}
            
        success_rates = [p.success_rate for p in performances]
        execution_times = [p.execution_time for p in performances]
        
        return {
            'success_trend': np.polyfit(range(len(success_rates)), success_rates, 1)[0],
            'time_trend': np.polyfit(range(len(execution_times)), execution_times, 1)[0],
            'stability': np.std(success_rates)
        }
        
    def _analyze_context_impact(self, 
                              performances: List[StrategyPerformance]) -> Dict[str, float]:
        """Analyze impact of context on performance."""
        if not performances:
            return {}
            
        context_scores = {}
        for perf in performances:
            for context, score in perf.context_performance.items():
                if context not in context_scores:
                    context_scores[context] = []
                context_scores[context].append(score)
                
        return {
            context: np.mean(scores)
            for context, scores in context_scores.items()
        }

class StrategyRefinement:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.refinement_history = {}
        
    def refine_strategy(self, strategy: str, 
                       performance_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Refine strategy based on performance analysis."""
        # Use LLM to suggest refinements
        refinement = self.llm.analyze_pattern({
            'strategy': strategy,
            'analysis': performance_analysis,
            'task': 'refinement'
        })
        
        result = {
            'refinements': refinement.text.split('\n'),
            'confidence': refinement.confidence,
            'priority': self._calculate_priority(performance_analysis)
        }
        
        self.refinement_history[strategy] = result
        return result
        
    def _calculate_priority(self, 
                          performance_analysis: Dict[str, Any]) -> float:
        """Calculate priority of refinements."""
        trends = performance_analysis['trends']
        return (0.4 * (1.0 - trends['success_trend']) +
                0.3 * trends['time_trend'] +
                0.3 * trends['stability'])

class ContextAdaptation:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.adaptation_history = {}
        
    def adapt_to_context(self, strategy: str, 
                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt strategy to specific context."""
        # Use LLM to suggest adaptations
        adaptation = self.llm.analyze_pattern({
            'strategy': strategy,
            'context': context,
            'task': 'adaptation'
        })
        
        result = {
            'adaptations': adaptation.text.split('\n'),
            'confidence': adaptation.confidence,
            'impact': self._estimate_impact(strategy, context)
        }
        
        self.adaptation_history[strategy] = result
        return result
        
    def _estimate_impact(self, strategy: str, 
                        context: Dict[str, Any]) -> Dict[str, float]:
        """Estimate impact of adaptations."""
        # Use LLM to estimate impact
        analysis = self.llm.analyze_pattern({
            'strategy': strategy,
            'context': context,
            'task': 'impact_estimation'
        })
        
        return {
            'effectiveness': analysis.confidence,
            'risk': 1.0 - analysis.confidence
        }
