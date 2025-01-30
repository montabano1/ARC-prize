from typing import Dict, Any, List, Optional
import numpy as np
import json
import time
from src.dsl.synthesizer import DSLProgram, DSLPrimitive

class PerformanceOptimizer:
    """Optimizes program performance based on results and LLM feedback"""
    def __init__(self, llm=None, primitive_library=None):
        self.llm = llm
        self.primitive_library = primitive_library
        self.optimization_history = []
        self.knowledge_base = {
            'successful_patterns': [],  # Patterns that worked well
            'failed_patterns': [],      # Patterns that didn't work
            'learned_strategies': [],   # Strategies that improved performance
            'pattern_success_rates': {},  # Track success rates for different pattern types
            'primitive_effectiveness': {},  # Track how well each primitive works
            'strategy_effectiveness': {}    # Track how well each strategy works
        }
        
    def optimize_program(self, program: DSLProgram, 
                        task_data: Dict[str, Any],
                        task_features: Dict[str, Any],
                        current_success_rate: float) -> DSLProgram:
        """Optimize a program using LLM feedback and learning"""
        if current_success_rate >= 0.9:  # High success rate
            self._record_success(program, task_features)
            return program
        else:
            self._record_failure(program, task_features, current_success_rate)
            
        # Convert task features to dict for JSON serialization
        task_features_dict = {
            'grid_size': task_features.grid_size,
            'unique_colors': task_features.unique_colors,
            'pattern_complexity': task_features.pattern_complexity,
            'overall_difficulty': task_features.overall_difficulty,
            'identified_patterns': task_features.identified_patterns,
            'confidence_scores': task_features.confidence_scores,
            'unified_strategy': task_features.unified_strategy,
            'best_primitives': self._get_best_primitives(task_features)
        }
            
        # Get LLM feedback on program
        feedback = self._get_llm_feedback(program, task_data, task_features_dict, current_success_rate)
        
        if feedback.get('suggest_new_primitives', False) and self.primitive_library:
            # Learn new primitives
            performance_data = {
                'success_rate': current_success_rate,
                'program_complexity': program.complexity,
                'failure_patterns': feedback.get('failure_patterns', [])
            }
            new_primitives = self.primitive_library.learn_new_primitives(
                task_features_dict, performance_data
            )
            
            if new_primitives:
                # Try to generate new program with new primitives
                optimized = self._regenerate_program(program, new_primitives, task_features_dict)
                if optimized:
                    program = optimized
        
        # Try optimization strategies
        strategies = feedback.get('optimization_strategies', [])
        for strategy in strategies:
            optimized = self._apply_optimization_strategy(program, strategy)
            if optimized:
                new_success = self._evaluate_program(optimized, task_data)
                if new_success > current_success_rate:
                    program = optimized
                    current_success_rate = new_success
                    
        # Record optimization attempt
        self.optimization_history.append({
            'original_success': current_success_rate,
            'optimized_success': self._evaluate_program(program, task_data),
            'feedback': feedback,
            'program_steps': [{'primitive': s['primitive'], 'params': s['params']} for s in program.steps]
        })
        
        return program
        
    def _get_llm_feedback(self, program: DSLProgram, 
                         task_data: Dict[str, Any],
                         task_features: Dict[str, Any],
                         current_success_rate: float) -> Dict[str, Any]:
        """Get LLM feedback on program performance"""
        if not self.llm:
            return {}
            
        prompt = (
            f"Analyze this program's performance and suggest improvements:\n\n"
            f"Program:\n{self._format_program(program)}\n\n"
            f"Task Features:\n{json.dumps(task_features, indent=2)}\n\n"
            f"Current Success Rate: {current_success_rate}\n\n"
            f"Return analysis in this JSON format:\n"
            f"{{\n"
            f"  'failure_analysis': 'What might be causing the failures',\n"
            f"  'suggest_new_primitives': true/false,\n"
            f"  'failure_patterns': ['pattern1', 'pattern2'],\n"
            f"  'optimization_strategies': [\n"
            f"    {{\n"
            f"      'type': 'strategy_type',\n"
            f"      'description': 'what to do',\n"
            f"      'steps': ['step1', 'step2']\n"
            f"    }}\n"
            f"  ]\n"
            f"}}"
        )
        
        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[
                {"role": "system", "content": "You are an expert at analyzing and improving visual pattern transformation programs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {}
            
    def _apply_optimization_strategy(self, program: DSLProgram, 
                                   strategy: Dict[str, Any]) -> Optional[DSLProgram]:
        """Apply an optimization strategy to a program"""
        if strategy['type'] == 'reorder':
            return self._reorder_operations(program)
        elif strategy['type'] == 'combine':
            return self._combine_operations(program)
        elif strategy['type'] == 'split':
            return self._split_operations(program)
        return None
        
    def _record_success(self, program: DSLProgram, task_features: Dict[str, Any]):
        """Record successful patterns and strategies"""
        # Record the pattern
        pattern_key = f"{task_features['pattern_complexity']}_{task_features['overall_difficulty']}"
        self.knowledge_base['successful_patterns'].append({
            'pattern': task_features['identified_patterns'],
            'strategy': task_features['unified_strategy'],
            'program_steps': program.steps,
            'complexity': program.complexity
        })
        
        # Update pattern success rates
        if pattern_key not in self.knowledge_base['pattern_success_rates']:
            self.knowledge_base['pattern_success_rates'][pattern_key] = []
        self.knowledge_base['pattern_success_rates'][pattern_key].append(1.0)
        
        # Update primitive effectiveness
        for step in program.steps:
            primitive = step['primitive']
            if primitive not in self.knowledge_base['primitive_effectiveness']:
                self.knowledge_base['primitive_effectiveness'][primitive] = []
            self.knowledge_base['primitive_effectiveness'][primitive].append(1.0)
            
    def _record_failure(self, program: DSLProgram, task_features: Dict[str, Any], success_rate: float):
        """Record failed patterns and update effectiveness metrics"""
        # Record the pattern
        pattern_key = f"{task_features['pattern_complexity']}_{task_features['overall_difficulty']}"
        self.knowledge_base['failed_patterns'].append({
            'pattern': task_features['identified_patterns'],
            'strategy': task_features['unified_strategy'],
            'program_steps': program.steps,
            'success_rate': success_rate
        })
        
        # Update pattern success rates
        if pattern_key not in self.knowledge_base['pattern_success_rates']:
            self.knowledge_base['pattern_success_rates'][pattern_key] = []
        self.knowledge_base['pattern_success_rates'][pattern_key].append(success_rate)
        
        # Update primitive effectiveness
        for step in program.steps:
            primitive = step['primitive']
            if primitive not in self.knowledge_base['primitive_effectiveness']:
                self.knowledge_base['primitive_effectiveness'][primitive] = []
            self.knowledge_base['primitive_effectiveness'][primitive].append(success_rate)
            
    def _get_best_primitives(self, task_features: Dict[str, Any]) -> List[str]:
        """Get most effective primitives for similar patterns"""
        pattern_key = f"{task_features['pattern_complexity']}_{task_features['overall_difficulty']}"
        
        # Get primitives that worked well for similar patterns
        effective_primitives = []
        for primitive, success_rates in self.knowledge_base['primitive_effectiveness'].items():
            if success_rates:
                avg_success = sum(success_rates) / len(success_rates)
                if avg_success > 0.7:  # Primitive has good track record
                    effective_primitives.append((primitive, avg_success))
                    
        return [p[0] for p in sorted(effective_primitives, key=lambda x: x[1], reverse=True)]
        
    def _regenerate_program(self, original_program: DSLProgram,
                           new_primitives: List[DSLPrimitive],
                           task_features: Dict[str, Any]) -> Optional[DSLProgram]:
        """Try to generate a new program using new primitives"""
        if not self.llm:
            return None
            
        prompt = (
            f"Given these new primitive operations:\n"
            + "\n".join(f"- {p.name}: {p.description}" for p in new_primitives)
            + f"\n\nAnd this original program:\n{self._format_program(original_program)}\n\n"
            f"Suggest a new program that uses the new primitives to solve the task more effectively.\n"
            f"Return the program in the same format as the original."
        )
        
        response = self.llm.client.chat.completions.create(
            model=self.llm.model,
            messages=[
                {"role": "system", "content": "You are an expert at optimizing visual pattern transformation programs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return DSLProgram(
                steps=result['steps'],
                complexity=self._calculate_complexity(result['steps']),
                description=result.get('description', "Regenerated program")
            )
        except Exception:
            return None
            
    def _format_program(self, program: DSLProgram) -> str:
        """Format program for LLM analysis"""
        steps = []
        for step in program.steps:
            steps.append({
                'primitive': step['primitive'],
                'params': step['params'],
                'description': step.get('description', '')
            })
        return json.dumps({'steps': steps}, indent=2)
        
    def _evaluate_program(self, program: DSLProgram, task_data: Dict[str, Any]) -> float:
        """Evaluate a program's success rate"""
        success = 0
        total = 0
        
        for example in task_data['train']:
            input_grid = np.array(example['input'])
            expected = np.array(example['output'])
            
            try:
                actual = self._execute_program(program, input_grid)
                if np.array_equal(actual, expected):
                    success += 1
            except Exception:
                pass
                
            total += 1
            
        return success / total if total > 0 else 0.0
        
    def _execute_program(self, program: DSLProgram, 
                        input_grid: np.ndarray) -> np.ndarray:
        """Execute a program on an input grid"""
        grid = input_grid.copy()
        for step in program.steps:
            # Execute each step (simplified for now)
            grid = self._apply_operation(grid, step)
        return grid
        
    def _apply_operation(self, grid: np.ndarray, 
                        operation: Dict[str, Any]) -> np.ndarray:
        """Apply an operation to a grid"""
        # Simplified operation application
        return grid
        
    def _reorder_operations(self, program: DSLProgram) -> DSLProgram:
        """Try reordering operations to improve performance"""
        # For now, just return the original program
        return program
        
    def _combine_operations(self, program: DSLProgram) -> DSLProgram:
        """Try combining similar consecutive operations"""
        # For now, just return the original program
        return program
        
    def _split_operations(self, program: DSLProgram) -> DSLProgram:
        """Try splitting complex operations into simpler ones"""
        # For now, just return the original program
        return program
        
    def _calculate_complexity(self, steps: List[Dict[str, Any]]) -> int:
        """Calculate the complexity of a program"""
        # For now, just return the number of steps
        return len(steps)
