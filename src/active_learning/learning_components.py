from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from src.llm.llm_interface import LLMInterface
import json
import time

@dataclass
class Hypothesis:
    id: str
    description: str
    confidence: float
    evidence: List[Dict[str, Any]]
    validation_results: Dict[str, Any]

class ConfidenceEstimator:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
    def estimate_confidence(self, hypothesis: Hypothesis, 
                          evidence: List[Dict[str, Any]]) -> Dict[str, float]:
        """Estimate confidence in a hypothesis."""
        # Use LLM to estimate confidence
        analysis = self.llm.analyze_pattern({
            'hypothesis': hypothesis,
            'evidence': evidence,
            'analysis_type': 'confidence'
        })
        
        return {
            'overall': analysis.confidence,
            'evidence_support': self._calculate_evidence_support(evidence),
            'generalization': self._estimate_generalization(hypothesis),
            'robustness': self._estimate_robustness(hypothesis, evidence)
        }
        
    def _calculate_evidence_support(self, evidence: List[Dict[str, Any]]) -> float:
        """Calculate how well evidence supports the hypothesis."""
        if not evidence:
            return 0.0
            
        support_scores = []
        for e in evidence:
            # Use LLM to analyze evidence
            analysis = self.llm.analyze_pattern({
                'evidence': e,
                'analysis_type': 'evidence_support'
            })
            support_scores.append(analysis.confidence)
            
        return np.mean(support_scores)
        
    def _estimate_generalization(self, hypothesis: Hypothesis) -> float:
        """Estimate how well the hypothesis generalizes."""
        # Use LLM to estimate generalization
        analysis = self.llm.analyze_pattern({
            'hypothesis': hypothesis,
            'analysis_type': 'generalization'
        })
        return analysis.confidence
        
    def _estimate_robustness(self, hypothesis: Hypothesis, 
                           evidence: List[Dict[str, Any]]) -> float:
        """Estimate robustness of the hypothesis."""
        # Use LLM to estimate robustness
        analysis = self.llm.analyze_pattern({
            'hypothesis': hypothesis,
            'evidence': evidence,
            'analysis_type': 'robustness'
        })
        return analysis.confidence

class TaskDifficultyAnalyzer:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
    def analyze_difficulty(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task difficulty comprehensively."""
        # Use LLM to analyze difficulty
        analysis = self.llm.analyze_pattern({
            'task': task,
            'analysis_type': 'task_difficulty'
        })
        
        return {
            'overall_difficulty': float(analysis.text.split('\n')[0]),
            'components': self._analyze_component_difficulty(task),
            'bottlenecks': self._identify_bottlenecks(task),
            'prerequisites': analysis.text.split('\n')[1:]
        }
        
    def _analyze_component_difficulty(self, task: Dict[str, Any]) -> Dict[str, float]:
        """Analyze difficulty of individual components."""
        components = {}
        for component in task.get('components', []):
            # Use LLM to analyze component
            analysis = self.llm.analyze_pattern({
                'component': component,
                'analysis_type': 'component_difficulty'
            })
            components[component['id']] = float(analysis.text)
        return components
        
    def _identify_bottlenecks(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential bottlenecks in task solution."""
        # Use LLM to identify bottlenecks
        analysis = self.llm.analyze_pattern({
            'task': task,
            'analysis_type': 'bottlenecks'
        })
        
        return [
            {'bottleneck': b, 'severity': float(s)}
            for b, s in [line.split(':') for line in analysis.text.split('\n')]
        ]

class HypothesisFormer:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
    async def form_hypothesis(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Form a hypothesis about the pattern based on examples."""
        try:
            # Analyze pattern using LLM
            prompt = f"""
            Analyze these examples and identify the pattern:
            {examples}
            
            Describe:
            1. The transformation pattern
            2. Key rules or constraints
            3. Any special cases or exceptions
            
            Respond with a valid JSON object containing:
            - pattern_description: A clear description of the pattern
            - rules: List of rules that define the pattern
            - confidence: A score between 0.0 and 1.0
            """
            
            response = await self.llm.get_completion(prompt)
            
            try:
                import json
                # Look for JSON-like content between curly braces
                start = response.find('{')
                end = response.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    result = json.loads(json_str)
                    
                    return {
                        'description': result.get('pattern_description', 'Failed to extract pattern description'),
                        'rules': result.get('rules', []),
                        'confidence': result.get('confidence', 0.0),
                        'examples': examples
                    }
            except json.JSONDecodeError:
                print("Failed to parse hypothesis result")
                
            return {
                'description': 'Failed to form hypothesis',
                'rules': [],
                'confidence': 0.0,
                'examples': examples
            }
            
        except Exception as e:
            print(f"Error forming hypothesis: {str(e)}")
            return {
                'description': f'Error: {str(e)}',
                'rules': [],
                'confidence': 0.0,
                'examples': examples
            }
            
    async def generate_test_cases(self, hypothesis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases to validate a hypothesis."""
        try:
            prompt = f"""
            Given this hypothesis about a pattern:
            {hypothesis['description']}
            
            Generate 3 test cases that would help validate this hypothesis.
            Each test case should have:
            1. An input grid
            2. Expected output grid
            3. Explanation of why this test case is useful
            
            Return a JSON array of test cases.
            """
            
            response = await self.llm.get_completion(prompt)
            
            try:
                import json
                # Look for JSON-like content between square brackets
                start = response.find('[')
                end = response.rfind(']') + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    test_cases = json.loads(json_str)
                    if isinstance(test_cases, list):
                        return test_cases
            except:
                print("Failed to parse test cases from response")
            
            return []
            
        except Exception as e:
            print(f"Error generating test cases: {str(e)}")
            return []

class LearningProgressTracker:
    """Tracks learning progress over time."""

    def __init__(self, llm: LLMInterface):
        """Initialize progress tracker."""
        self.llm = llm
        self.history = []
        
    async def track_progress(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track learning progress from events."""
        try:
            # Add events to history
            self.history.extend(events)
            
            # Calculate basic metrics
            total_events = len(self.history)
            if not total_events:
                return {
                    "overall_progress": 0.0,
                    "metrics": {},
                    "insights": []
                }
                
            successful_events = sum(1 for e in self.history if e.get('success', False))
            overall_progress = successful_events / total_events if total_events > 0 else 0
            
            # Create progress summary
            progress = {
                "overall_progress": overall_progress,
                "metrics": {
                    "total_tasks": total_events,
                    "successful_tasks": successful_events,
                    "success_rate": overall_progress
                },
                "insights": [
                    "Making steady progress in pattern recognition",
                    "Need more practice with complex transformations"
                ]
            }
            
            return progress
            
        except Exception as e:
            print(f"Error tracking progress: {str(e)}")
            return {
                "overall_progress": 0.0,
                "metrics": {},
                "insights": [f"Error tracking progress: {str(e)}"]
            }

class PatternValidator:
    """Validates patterns against examples."""
    
    def __init__(self, llm: LLMInterface):
        """Initialize pattern validator."""
        self.llm = llm
        
    async def validate_pattern(self, pattern: Dict[str, Any], examples: Dict[str, Any]) -> Dict[str, bool]:
        """Validate a pattern against examples."""
        try:
            # Format examples for validation
            examples_str = ""
            if 'train' in examples:
                for i, example in enumerate(examples['train'], 1):
                    examples_str += f"\nExample {i}:\n"
                    examples_str += f"Input:\n{self._format_grid(example['input'])}\n"
                    examples_str += f"Output:\n{self._format_grid(example['output'])}\n"
            
            # Create validation prompt
            prompt = f"""
            Validate this pattern against the examples:
            
            Pattern:
            {pattern.get('description', 'No description available')}
            
            Examples:
            {examples_str}
            
            Return a JSON object with:
            {{
                "is_valid": true/false,
                "confidence": 0.0-1.0,
                "explanation": "why valid/invalid"
            }}
            """
            
            # Get LLM validation
            response = await self.llm.get_completion(prompt)
            
            # Parse response
            try:
                import json
                result = json.loads(response)
                return result
            except:
                return {
                    "is_valid": False,
                    "confidence": 0.0,
                    "explanation": "Failed to validate pattern"
                }
                
        except Exception as e:
            print(f"Error validating pattern: {str(e)}")
            return {
                "is_valid": False,
                "confidence": 0.0,
                "explanation": str(e)
            }
            
    def _format_grid(self, grid: List[List[int]]) -> str:
        """Format grid for prompt."""
        return '\n'.join(' '.join(str(x) for x in row) for row in grid)

class StrategyEffectivenessMonitor:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.effectiveness_history = {}
        
    def monitor_effectiveness(self, strategy: Dict[str, Any], 
                            results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Monitor effectiveness of a strategy."""
        # Use LLM to analyze effectiveness
        analysis = self.llm.analyze_pattern({
            'strategy': strategy,
            'results': results,
            'analysis_type': 'effectiveness'
        })
        
        effectiveness = {
            'overall_score': float(analysis.text.split('\n')[0]),
            'metrics': self._calculate_metrics(results),
            'strengths': self._identify_strengths(strategy, results),
            'weaknesses': self._identify_weaknesses(strategy, results)
        }
        
        self.effectiveness_history[strategy['id']] = effectiveness
        return effectiveness
        
    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate effectiveness metrics."""
        return {
            'success_rate': sum(1 for r in results if r.get('success', False)) / len(results),
            'avg_confidence': np.mean([r.get('confidence', 0.0) for r in results]),
            'avg_time': np.mean([r.get('time', 0.0) for r in results])
        }
        
    def _identify_strengths(self, strategy: Dict[str, Any], 
                          results: List[Dict[str, Any]]) -> List[str]:
        """Identify strategy strengths."""
        # Use LLM to identify strengths
        analysis = self.llm.analyze_pattern({
            'strategy': strategy,
            'results': results,
            'analysis_type': 'strengths'
        })
        return analysis.text.split('\n')
        
    def _identify_weaknesses(self, strategy: Dict[str, Any], 
                           results: List[Dict[str, Any]]) -> List[str]:
        """Identify strategy weaknesses."""
        # Use LLM to identify weaknesses
        analysis = self.llm.analyze_pattern({
            'strategy': strategy,
            'results': results,
            'analysis_type': 'weaknesses'
        })
        return analysis.text.split('\n')
