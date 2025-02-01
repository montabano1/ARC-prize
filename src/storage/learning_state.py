import json
import os
import time
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)

class LearningStateManager:
    """Manages persistent learning state across runs"""
    
    def __init__(self, storage_dir: str = "data/learning_state"):
        self.storage_dir = storage_dir
        self.state_file = os.path.join(storage_dir, "learning_state.json")
        self.history_dir = os.path.join(storage_dir, "history")
        self.concepts_dir = os.path.join(storage_dir, "concepts")
        self.strategies_dir = os.path.join(storage_dir, "strategies")
        self.meta_insights_dir = os.path.join(storage_dir, "meta_insights")
        self.patterns_dir = os.path.join(storage_dir, "patterns")
        
        # Create directories if they don't exist
        for directory in [self.storage_dir, self.history_dir, 
                         self.concepts_dir, self.strategies_dir, 
                         self.meta_insights_dir, self.patterns_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Initialize or load state
        self.state = self._load_state()
        
    def _load_state(self) -> Dict[str, Any]:
        """Load or initialize learning state"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {
            'version': '1.0',
            'last_updated': datetime.now().isoformat(),
            'runs_count': 0,
            'successful_patterns': {},
            'failed_patterns': {},
            'concept_evolution': {},
            'strategy_effectiveness': {},
            'meta_learning_insights': []
        }
    
    def save_state(self):
        """Save current state to disk"""
        self.state['last_updated'] = datetime.now().isoformat()
        
        # Create backup
        if os.path.exists(self.state_file):
            backup_file = f"{self.state_file}.bak"
            shutil.copy2(self.state_file, backup_file)
        
        # Save new state
        try:
            with open(self.state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            # Restore backup if save failed
            if os.path.exists(f"{self.state_file}.bak"):
                shutil.copy2(f"{self.state_file}.bak", self.state_file)
                
    def update_state(self, updates: Dict[str, Any]):
        """Update specific parts of the state."""
        for key, value in updates.items():
            if key in self.state:
                if isinstance(self.state[key], list):
                    if isinstance(value, list):
                        self.state[key].extend(value)
                    else:
                        self.state[key].append(value)
                else:
                    self.state[key] = value
        
    def record_run(self, run_data: Dict[str, Any]):
        """Record a learning run with comprehensive metadata"""
        timestamp = datetime.now().isoformat()
        run_id = f"run_{self.state['runs_count'] + 1}"
        
        # Save run data to history
        run_file = os.path.join(self.history_dir, f"{run_id}.json")
        with open(run_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'run_id': run_id,
                'data': run_data
            }, f, indent=2)
            
        self.state['runs_count'] += 1
        self.save_state()
        
    def record_strategy(self, strategy_data: Dict[str, Any]):
        """Record a successful strategy with its context and performance"""
        timestamp = datetime.now().isoformat()
        strategy_id = strategy_data.get('id', f"strategy_{len(os.listdir(self.strategies_dir))}")
        
        # Update effectiveness tracking
        if strategy_id not in self.state['strategy_effectiveness']:
            self.state['strategy_effectiveness'][strategy_id] = {
                'successes': 0,
                'attempts': 0,
                'contexts': []
            }
        
        stats = self.state['strategy_effectiveness'][strategy_id]
        stats['attempts'] += 1
        if strategy_data.get('success', False):
            stats['successes'] += 1
        stats['contexts'].append(strategy_data.get('context', {}))
        
        # Save strategy details
        strategy_file = os.path.join(self.strategies_dir, f"{strategy_id}.json")
        with open(strategy_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'strategy_id': strategy_id,
                'data': strategy_data,
                'stats': stats
            }, f, indent=2)
            
        self.save_state()
        
    def record_concept(self, concept_data: Dict[str, Any]):
        """Record a learned concept with its examples and rules"""
        timestamp = datetime.now().isoformat()
        concept_id = concept_data.get('id', f"concept_{len(os.listdir(self.concepts_dir))}")
        
        # Track concept evolution
        if concept_id not in self.state['concept_evolution']:
            self.state['concept_evolution'][concept_id] = {
                'versions': [],
                'refinements': 0
            }
            
        evolution = self.state['concept_evolution'][concept_id]
        evolution['versions'].append({
            'timestamp': timestamp,
            'confidence': concept_data.get('confidence', 0.0)
        })
        evolution['refinements'] += 1
        
        # Save concept details
        concept_file = os.path.join(self.concepts_dir, f"{concept_id}.json")
        with open(concept_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'concept_id': concept_id,
                'data': concept_data,
                'evolution': evolution
            }, f, indent=2)
            
        self.save_state()
        
    def record_pattern(self, pattern_data: Dict[str, Any], success: bool = True):
        """Record a pattern and its effectiveness"""
        timestamp = datetime.now().isoformat()
        pattern_id = pattern_data.get('id', f"pattern_{len(os.listdir(self.patterns_dir))}")
        
        # Update pattern tracking
        target_dict = 'successful_patterns' if success else 'failed_patterns'
        if pattern_id not in self.state[target_dict]:
            self.state[target_dict][pattern_id] = {
                'occurrences': 0,
                'contexts': []
            }
            
        pattern_stats = self.state[target_dict][pattern_id]
        pattern_stats['occurrences'] += 1
        pattern_stats['contexts'].append(pattern_data.get('context', {}))
        
        # Save pattern details
        pattern_file = os.path.join(self.patterns_dir, f"{pattern_id}.json")
        with open(pattern_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'pattern_id': pattern_id,
                'success': success,
                'data': pattern_data,
                'stats': pattern_stats
            }, f, indent=2)
            
        self.save_state()
        
    def record_meta_insight(self, insight: str, importance: float):
        """Record a meta-learning insight"""
        timestamp = datetime.now().isoformat()
        
        insight_data = {
            'timestamp': timestamp,
            'insight': insight,
            'importance': importance
        }
        
        self.state['meta_learning_insights'].append(insight_data)
        
        # Save insight to file
        insight_file = os.path.join(
            self.meta_insights_dir, 
            f"insight_{len(self.state['meta_learning_insights'])}.json"
        )
        with open(insight_file, 'w') as f:
            json.dump(insight_data, f, indent=2)
            
        self.save_state()
        
    def get_relevant_patterns(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve patterns relevant to the given context"""
        relevant_patterns = []
        
        # Load all pattern files
        for pattern_file in os.listdir(self.patterns_dir):
            if not pattern_file.endswith('.json'):
                continue
                
            with open(os.path.join(self.patterns_dir, pattern_file), 'r') as f:
                pattern_data = json.load(f)
                
            # Check if pattern matches context
            if self._pattern_matches_context(pattern_data['data'], context):
                relevant_patterns.append(pattern_data)
                
        return relevant_patterns
        
    def get_effective_strategies(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Retrieve strategies that have been effective in similar contexts"""
        effective_strategies = []
        
        # Load all strategy files
        for strategy_file in os.listdir(self.strategies_dir):
            if not strategy_file.endswith('.json'):
                continue
                
            with open(os.path.join(self.strategies_dir, strategy_file), 'r') as f:
                strategy_data = json.load(f)
                
            # Check strategy effectiveness
            stats = strategy_data['stats']
            if stats['attempts'] > 0:
                success_rate = stats['successes'] / stats['attempts']
                if success_rate > 0.7:  # Consider strategies with >70% success rate
                    effective_strategies.append(strategy_data)
                    
        return effective_strategies
        
    def _pattern_matches_context(self, pattern: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check if a pattern matches the given context"""
        # Implement pattern matching logic here
        # For now, just check if pattern has any matching keys with context
        return bool(set(pattern.keys()) & set(context.keys()))
        
    def get_state(self) -> Dict[str, Any]:
        """Get the current learning state."""
        return self.state
        
    def get_recent_runs(self, n: int) -> List[Dict[str, Any]]:
        """Get the N most recent task runs."""
        history = self.state.get('performance_history', [])
        return history[-n:] if history else []
        
    def get_successful_strategies(self) -> List[str]:
        """Get strategies that have been successful."""
        strategies = self.state.get('strategies', {})
        return [
            name for name, data in strategies.items()
            if data.get('success_rate', 0.0) > 0.7
        ]
        
    def get_concepts_by_confidence(self, min_confidence: float) -> List[str]:
        """Get concepts that have at least the specified confidence level."""
        concepts = self.state.get('concepts', {})
        return [
            name for name, data in concepts.items()
            if data.get('confidence', 0.0) >= min_confidence
        ]
        
    def record_task_attempt(self, task_id: str, success: bool, strategy: Optional[str] = None) -> None:
        """Record an attempt at solving a task."""
        history = self.state.get('performance_history', [])
        history.append({
            'task_id': task_id,
            'success': success,
            'strategy': strategy,
            'timestamp': datetime.now().isoformat()
        })
        self.state['performance_history'] = history
        
        # Update strategy success rate if used
        if strategy:
            strategies = self.state.get('strategies', {})
            if strategy not in strategies:
                strategies[strategy] = {'attempts': 0, 'successes': 0}
            strategies[strategy]['attempts'] += 1
            if success:
                strategies[strategy]['successes'] += 1
            strategies[strategy]['success_rate'] = (
                strategies[strategy]['successes'] / 
                strategies[strategy]['attempts']
            )
            self.state['strategies'] = strategies
            
    def update_concept_confidence(self, concept: str, confidence: float) -> None:
        """Update the confidence level for a concept."""
        concepts = self.state.get('concepts', {})
        if concept not in concepts:
            concepts[concept] = {}
        concepts[concept]['confidence'] = confidence
        self.state['concepts'] = concepts
        
    def add_available_task(self, task: Dict[str, Any]) -> None:
        """Add a task to the available tasks list."""
        tasks = self.state.get('available_tasks', [])
        if task not in tasks:
            tasks.append(task)
        self.state['available_tasks'] = tasks
        
    def remove_available_task(self, task_id: str) -> None:
        """Remove a task from the available tasks list."""
        tasks = self.state.get('available_tasks', [])
        self.state['available_tasks'] = [
            t for t in tasks if t.get('id') != task_id
        ]
