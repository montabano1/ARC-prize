from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np
from src.llm.llm_interface import LLMInterface

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
        
    def form_hypothesis(self, observations: List[Dict[str, Any]]) -> Hypothesis:
        """Form a hypothesis based on observations."""
        # Use LLM to form hypothesis
        analysis = self.llm.analyze_pattern({
            'observations': observations,
            'analysis_type': 'hypothesis_formation'
        })
        
        hypothesis = Hypothesis(
            id=f"hyp_{len(observations)}",
            description=analysis.text.split('\n')[0],
            confidence=analysis.confidence,
            evidence=observations,
            validation_results={}
        )
        
        # Validate hypothesis
        hypothesis.validation_results = self._validate_hypothesis(hypothesis)
        return hypothesis
        
    def _validate_hypothesis(self, hypothesis: Hypothesis) -> Dict[str, Any]:
        """Validate formed hypothesis."""
        # Use LLM to validate hypothesis
        validation = self.llm.validate_concept(
            hypothesis.description,
            hypothesis.evidence
        )
        
        return {
            'is_valid': validation.confidence > 0.7,
            'confidence': validation.confidence,
            'issues': self._identify_issues(hypothesis)
        }
        
    def _identify_issues(self, hypothesis: Hypothesis) -> List[str]:
        """Identify potential issues with the hypothesis."""
        # Use LLM to identify issues
        analysis = self.llm.analyze_pattern({
            'hypothesis': hypothesis,
            'analysis_type': 'issues'
        })
        
        return analysis.text.split('\n')

class PatternValidator:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        
    def validate_pattern(self, pattern: Dict[str, Any], 
                        examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate a pattern against examples."""
        # Use LLM to validate pattern
        validation = self.llm.validate_concept(
            pattern['description'],
            examples
        )
        
        return {
            'is_valid': validation.confidence > 0.7,
            'confidence': validation.confidence,
            'matches': self._find_pattern_matches(pattern, examples),
            'exceptions': self._find_exceptions(pattern, examples)
        }
        
    def _find_pattern_matches(self, pattern: Dict[str, Any], 
                            examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find examples that match the pattern."""
        matches = []
        for example in examples:
            # Use LLM to check match
            analysis = self.llm.analyze_pattern({
                'pattern': pattern,
                'example': example,
                'analysis_type': 'pattern_match'
            })
            
            if analysis.confidence > 0.7:
                matches.append({
                    'example': example,
                    'confidence': analysis.confidence
                })
                
        return matches
        
    def _find_exceptions(self, pattern: Dict[str, Any], 
                        examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find examples that are exceptions to the pattern."""
        exceptions = []
        for example in examples:
            # Use LLM to check exception
            analysis = self.llm.analyze_pattern({
                'pattern': pattern,
                'example': example,
                'analysis_type': 'pattern_exception'
            })
            
            if analysis.confidence > 0.7:
                exceptions.append({
                    'example': example,
                    'reason': analysis.text
                })
                
        return exceptions

class LearningProgressTracker:
    def __init__(self, llm: LLMInterface):
        self.llm = llm
        self.progress_history = {}
        
    def track_progress(self, learning_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Track learning progress over time."""
        # Use LLM to analyze progress
        analysis = self.llm.analyze_pattern({
            'events': learning_events,
            'analysis_type': 'learning_progress'
        })
        
        progress = {
            'overall_progress': float(analysis.text.split('\n')[0]),
            'milestones': self._identify_milestones(learning_events),
            'trends': self._analyze_trends(learning_events),
            'recommendations': analysis.text.split('\n')[1:]
        }
        
        self.progress_history[len(self.progress_history)] = progress
        return progress
        
    def _identify_milestones(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify significant learning milestones."""
        milestones = []
        for i, event in enumerate(events):
            # Use LLM to check milestone
            analysis = self.llm.analyze_pattern({
                'event': event,
                'history': events[:i],
                'analysis_type': 'milestone'
            })
            
            if analysis.confidence > 0.7:
                milestones.append({
                    'event': event,
                    'significance': analysis.text
                })
                
        return milestones
        
    def _analyze_trends(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze learning trends."""
        if not events:
            return {}
            
        # Extract metrics for trend analysis
        metrics = {
            'success_rate': [e.get('success', False) for e in events],
            'confidence': [e.get('confidence', 0.0) for e in events],
            'complexity': [e.get('complexity', 0.0) for e in events]
        }
        
        return {
            name: {
                'trend': np.polyfit(range(len(values)), values, 1)[0],
                'mean': np.mean(values),
                'std': np.std(values)
            }
            for name, values in metrics.items()
        }

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
