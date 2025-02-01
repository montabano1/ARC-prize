from typing import Dict, Any, List
import numpy as np
from src.llm.llm_interface import LLMInterface

class PerformanceOptimizer:
    """Optimizes strategy performance."""
    
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
    async def optimize_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize a strategy for better performance."""
        try:
            # Analyze current strategy
            analysis = await self._analyze_strategy(strategy)
            
            # Generate optimizations
            optimizations = await self._generate_optimizations(strategy, analysis)
            
            # Apply optimizations
            optimized_strategy = await self._apply_optimizations(strategy, optimizations)
            
            return optimized_strategy
            
        except Exception as e:
            print(f"Error optimizing strategy: {str(e)}")
            return strategy
            
    async def _analyze_strategy(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze strategy for optimization opportunities."""
        try:
            prompt = f"""
            Analyze this strategy for optimization opportunities:
            {strategy}
            
            Consider:
            1. Computational efficiency
            2. Pattern recognition accuracy
            3. Solution robustness
            
            Return a JSON object with:
            {{
                "efficiency_score": float (0-1),
                "accuracy_score": float (0-1),
                "robustness_score": float (0-1),
                "optimization_opportunities": [string]
            }}
            """
            
            response = await self.llm.get_completion(prompt)
            
            try:
                import json
                return json.loads(response)
            except json.JSONDecodeError:
                return {
                    "efficiency_score": 0.5,
                    "accuracy_score": 0.5,
                    "robustness_score": 0.5,
                    "optimization_opportunities": []
                }
                
        except Exception as e:
            print(f"Error analyzing strategy: {str(e)}")
            return {
                "efficiency_score": 0.5,
                "accuracy_score": 0.5,
                "robustness_score": 0.5,
                "optimization_opportunities": []
            }
            
    async def _generate_optimizations(self, strategy: Dict[str, Any], 
                                    analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate specific optimizations based on analysis."""
        try:
            prompt = f"""
            Generate optimizations for this strategy:
            Strategy: {strategy}
            Analysis: {analysis}
            
            Return a JSON array of optimizations, each with:
            {{
                "type": "efficiency|accuracy|robustness",
                "description": string,
                "implementation": {{
                    "before": string,
                    "after": string
                }}
            }}
            """
            
            response = await self.llm.get_completion(prompt)
            
            try:
                import json
                return json.loads(response)
            except json.JSONDecodeError:
                return []
                
        except Exception as e:
            print(f"Error generating optimizations: {str(e)}")
            return []
            
    async def _apply_optimizations(self, strategy: Dict[str, Any], 
                                 optimizations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply optimizations to strategy."""
        try:
            optimized = strategy.copy()
            
            for opt in optimizations:
                if opt['type'] == 'efficiency':
                    # Apply efficiency optimizations
                    if 'steps' in optimized:
                        optimized['steps'] = self._optimize_steps(optimized['steps'])
                elif opt['type'] == 'accuracy':
                    # Apply accuracy optimizations
                    if 'validation' in optimized:
                        optimized['validation']['threshold'] = max(
                            optimized['validation'].get('threshold', 0.8),
                            0.9
                        )
                elif opt['type'] == 'robustness':
                    # Apply robustness optimizations
                    if 'error_handling' not in optimized:
                        optimized['error_handling'] = {
                            'retry_count': 3,
                            'fallback_strategy': 'simple'
                        }
                        
            return optimized
            
        except Exception as e:
            print(f"Error applying optimizations: {str(e)}")
            return strategy
            
    def _optimize_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize execution steps."""
        try:
            # Remove redundant steps
            unique_steps = []
            seen = set()
            for step in steps:
                step_key = f"{step['type']}_{step.get('parameters', {})}"
                if step_key not in seen:
                    seen.add(step_key)
                    unique_steps.append(step)
                    
            # Reorder steps for efficiency
            optimized_steps = sorted(unique_steps, 
                                   key=lambda x: self._get_step_priority(x['type']))
                                   
            return optimized_steps
            
        except Exception as e:
            print(f"Error optimizing steps: {str(e)}")
            return steps
            
    def _get_step_priority(self, step_type: str) -> int:
        """Get execution priority for step type."""
        priorities = {
            'analyze': 1,
            'transform': 2,
            'validate': 3,
            'apply': 4
        }
        return priorities.get(step_type, 99)
