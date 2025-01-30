from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from src.meta_strategy.strategy_selector import StrategySelector
from src.llm.llm_interface import LLMInterface

@dataclass
class StrategyExecutionResult:
    success: bool
    output: Any
    performance_metrics: Dict[str, float]
    resource_usage: Dict[str, float]
    execution_time: float

class StrategyCoordinator:
    def __init__(self, strategy_selector: StrategySelector, llm: LLMInterface):
        self.strategy_selector = strategy_selector
        self.llm = llm
        self.active_strategies = {}
        self.performance_history = {}
        
    def coordinate_strategy_execution(self, task: Dict[str, Any]) -> StrategyExecutionResult:
        """Coordinate the execution of strategies for a task."""
        selected_strategy = self.strategy_selector.select_strategy(task)
        return self.execute_strategy(selected_strategy, task)
        
    def execute_strategy(self, strategy: str, task: Dict[str, Any]) -> StrategyExecutionResult:
        """Execute a selected strategy with resource management."""
        try:
            # Allocate resources
            resources = self.resource_allocator.allocate(strategy, task)
            
            # Execute strategy
            result = self.strategy_executor.execute(strategy, task, resources)
            
            # Monitor performance
            self.performance_monitor.update(strategy, result)
            
            # Update history
            self.update_performance_history(strategy, result)
            
            return result
            
        except Exception as e:
            # Switch to fallback strategy if needed
            return self.handle_strategy_failure(strategy, task, e)

class ResourceAllocator:
    def __init__(self):
        self.resource_pools = {
            'memory': 1.0,  # Available memory pool
            'compute': 1.0,  # Available compute resources
            'time': 1.0     # Available time budget
        }
        self.allocations = {}
        
    def allocate(self, strategy: str, requirements: Dict[str, float]) -> Dict[str, float]:
        """Allocate resources for a strategy execution."""
        allocation = {}
        for resource, amount in requirements.items():
            if resource not in self.resource_pools:
                raise ValueError(f"Unknown resource type: {resource}")
            if amount > self.resource_pools[resource]:
                raise ValueError(f"Insufficient {resource} resources")
            
            allocation[resource] = amount
            self.resource_pools[resource] -= amount
            
        self.allocations[strategy] = allocation
        return allocation
        
    def release(self, strategy: str):
        """Release resources allocated to a strategy."""
        if strategy in self.allocations:
            for resource, amount in self.allocations[strategy].items():
                self.resource_pools[resource] += amount
            del self.allocations[strategy]

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        self.thresholds = {
            'execution_time': 30.0,  # seconds
            'memory_usage': 0.8,     # percentage
            'success_rate': 0.7      # percentage
        }
        
    def update(self, strategy: str, result: StrategyExecutionResult):
        """Update performance metrics for a strategy."""
        if strategy not in self.metrics:
            self.metrics[strategy] = []
            
        self.metrics[strategy].append({
            'execution_time': result.execution_time,
            'success': result.success,
            'resource_usage': result.resource_usage
        })
        
        # Check if performance is below thresholds
        if self.check_performance_issues(strategy):
            self.trigger_performance_alert(strategy)
            
    def check_performance_issues(self, strategy: str) -> bool:
        """Check if there are performance issues with a strategy."""
        recent_metrics = self.metrics[strategy][-10:]  # Look at last 10 executions
        
        avg_time = np.mean([m['execution_time'] for m in recent_metrics])
        success_rate = np.mean([float(m['success']) for m in recent_metrics])
        max_memory = max([m['resource_usage'].get('memory', 0) for m in recent_metrics])
        
        return (avg_time > self.thresholds['execution_time'] or
                success_rate < self.thresholds['success_rate'] or
                max_memory > self.thresholds['memory_usage'])
                
    def trigger_performance_alert(self, strategy: str):
        """Trigger an alert when performance issues are detected."""
        # Implementation would depend on the alert system
        pass

class StrategySwitcher:
    def __init__(self, strategy_selector: StrategySelector):
        self.strategy_selector = strategy_selector
        self.switch_history = {}
        
    def should_switch_strategy(self, current_strategy: str, 
                             performance_metrics: Dict[str, float]) -> bool:
        """Determine if we should switch to a different strategy."""
        # Check performance thresholds
        if (performance_metrics['success_rate'] < 0.6 or
            performance_metrics['execution_time'] > 30.0):
            return True
            
        # Check if we're stuck in a local optimum
        if self.is_stuck(current_strategy, performance_metrics):
            return True
            
        return False
        
    def select_new_strategy(self, task: Dict[str, Any], 
                          excluded_strategy: str) -> str:
        """Select a new strategy, excluding the current one."""
        return self.strategy_selector.select_strategy(
            task, exclude=[excluded_strategy]
        )
        
    def is_stuck(self, strategy: str, metrics: Dict[str, float]) -> bool:
        """Check if the strategy is stuck in a local optimum."""
        if strategy not in self.switch_history:
            return False
            
        recent_switches = self.switch_history[strategy][-5:]
        if len(recent_switches) < 5:
            return False
            
        # Check if performance hasn't improved over last 5 switches
        return all(m['performance'] >= metrics['success_rate'] 
                  for m in recent_switches)

class ResultsAnalyzer:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.analysis_history = {}
        
    def analyze_results(self, strategy: str, 
                       results: List[StrategyExecutionResult]) -> Dict[str, Any]:
        """Analyze the results of strategy execution."""
        analysis = {
            'success_rate': self.calculate_success_rate(results),
            'performance_trends': self.analyze_performance_trends(results),
            'resource_efficiency': self.analyze_resource_usage(results),
            'failure_patterns': self.analyze_failure_patterns(results)
        }
        
        # Get LLM insights
        llm_analysis = self.get_llm_insights(strategy, analysis)
        analysis['llm_insights'] = llm_analysis
        
        self.analysis_history[strategy] = analysis
        return analysis
        
    def calculate_success_rate(self, 
                             results: List[StrategyExecutionResult]) -> float:
        """Calculate the success rate of strategy execution."""
        if not results:
            return 0.0
        return sum(1 for r in results if r.success) / len(results)
        
    def analyze_performance_trends(self, 
                                 results: List[StrategyExecutionResult]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        if not results:
            return {}
            
        execution_times = [r.execution_time for r in results]
        return {
            'mean_time': np.mean(execution_times),
            'trend': np.polyfit(range(len(execution_times)), execution_times, 1)[0],
            'variance': np.var(execution_times)
        }
        
    def analyze_resource_usage(self, 
                             results: List[StrategyExecutionResult]) -> Dict[str, float]:
        """Analyze resource usage efficiency."""
        if not results:
            return {}
            
        resource_types = results[0].resource_usage.keys()
        usage_stats = {}
        
        for resource in resource_types:
            usage = [r.resource_usage[resource] for r in results]
            usage_stats[resource] = {
                'mean': np.mean(usage),
                'max': max(usage),
                'min': min(usage)
            }
            
        return usage_stats
        
    def analyze_failure_patterns(self, 
                               results: List[StrategyExecutionResult]) -> List[Dict[str, Any]]:
        """Analyze patterns in strategy failures."""
        failures = [r for r in results if not r.success]
        if not failures:
            return []
            
        patterns = []
        # Group failures by similar characteristics
        # This would involve more sophisticated pattern matching
        return patterns
        
    def get_llm_insights(self, strategy: str, 
                        analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Get insights from LLM about the analysis."""
        prompt = self._create_analysis_prompt(strategy, analysis)
        response = self.llm.analyze_pattern(prompt)
        return {
            'insights': response.text,
            'confidence': response.confidence
        }
        
    def _create_analysis_prompt(self, strategy: str, 
                              analysis: Dict[str, Any]) -> str:
        """Create a prompt for LLM analysis."""
        return f"""Analyze the following strategy execution results:

        Strategy: {strategy}
        Success Rate: {analysis['success_rate']}
        Performance Trends: {analysis['performance_trends']}
        Resource Usage: {analysis['resource_efficiency']}

        Please provide:
        1. Key insights about the strategy's performance
        2. Potential areas for improvement
        3. Recommendations for optimization
        4. Risk factors to consider
        """
