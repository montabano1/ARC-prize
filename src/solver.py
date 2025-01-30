import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import os
import numpy as np

# Load environment variables from .env file
load_dotenv()

from src.llm.llm_interface import LLMInterface
from src.meta_strategy.strategy_selector import StrategySelector
from src.task_assessment.task_analyzer import TaskAnalyzer
from src.concept_formation.concept_extractor import ConceptExtractor
from src.pattern_graph.pattern_graph import PatternGraph
from src.dsl_engine.dsl_synthesizer import DSLSynthesizer
from src.integration.strategy_integration import StrategyCoordinator
from src.integration.concept_pattern_bridge import ConceptPatternMapper
from src.integration.solution_synthesis import ProgramComposer
from src.strategies.basic_strategies import STRATEGY_REGISTRY

class ARCSolver:
    def __init__(self, debug: bool = False):
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "system_config.yaml"
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup logging
        log_level = logging.DEBUG if debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI API key
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        # Initialize components
        self.llm = LLMInterface(openai_api_key)
        self.strategy_selector = StrategySelector(self.llm)
        self.task_analyzer = TaskAnalyzer(self.llm)
        self.concept_extractor = ConceptExtractor(self.llm)
        self.pattern_graph = PatternGraph()
        self.dsl_synthesizer = DSLSynthesizer()
        
        # Register basic strategies
        for name, strategy_info in STRATEGY_REGISTRY.items():
            self.strategy_selector.register_strategy(
                name=name,
                strategy_fn=strategy_info['function'],
                context_requirements=strategy_info['requirements']
            )
        
        # Initialize integration components
        self.strategy_coordinator = StrategyCoordinator(self.strategy_selector, self.llm)
        self.concept_pattern_mapper = ConceptPatternMapper(self.llm)
        self.program_composer = ProgramComposer(self.llm)
        
        # State tracking
        self.current_task = None
        self.current_solution = None
        self.performance_history = []
        
    def solve(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Solve an ARC task"""
        self.logger.info(f"Starting to solve task {task.get('task_id', 'unknown')}")
        self.current_task = task
        
        # Convert task data to numpy arrays
        train_data = [(np.array(pair['input']), np.array(pair['output']))
                     for pair in task['train']]
        test_data = [(np.array(pair['input']), np.array(pair['output']))
                    for pair in task['test']]
        
        # Task assessment
        assessment = self.task_analyzer.analyze_task(
            train_data[0][0], train_data[0][1]
        )
        self.logger.debug(f"Task assessment: {assessment}")
        
        # Get applicable concepts
        concepts = self.concept_extractor.extract_concepts(
            train_data[0][0], train_data[0][1]
        )
        self.logger.debug(f"Found {len(concepts)} applicable concepts")
        
        # Select strategy
        strategy = self.strategy_selector.select_strategy({
            'task': task,
            'assessment': assessment,
            'concepts': concepts,
            'confidence_threshold': self.config['meta_strategy']['confidence_threshold']
        })
        
        # Execute strategy
        solution = self.strategy_coordinator.coordinate_strategy_execution({
            'strategy': strategy,
            'task': task
        })
        
        self.current_solution = solution
        self._update_performance(solution)
        
        return solution
        
    def train(self, task: Dict[str, Any]):
        """Train on a task"""
        solution = self.solve(task)
        if 'output_grid' in task:
            success = self._validate_solution(solution, task['output_grid'])
            self._learn_from_execution(task, solution, success)
            
    def get_state(self) -> Dict[str, Any]:
        """Get current system state"""
        return {
            'current_task': self.current_task,
            'current_solution': self.current_solution,
            'performance': self._get_performance_metrics(),
            'concepts': self.concept_extractor.get_concepts(),
            'patterns': self.pattern_graph.get_patterns()
        }
        
    def explain_solution(self) -> str:
        """Get explanation for current solution"""
        if not self.current_solution:
            return "No solution available"
            
        return self.llm.generate_explanation(self.current_solution)
        
    def process_feedback(self, feedback: str):
        """Process user feedback"""
        if self.current_solution:
            self.strategy_selector.update_from_feedback(feedback)
            self.concept_extractor.update_from_feedback(feedback)
            
    def list_concepts(self) -> List[Dict[str, Any]]:
        """List all learned concepts"""
        return self.concept_extractor.get_concepts()
        
    def _validate_solution(self, solution: Dict[str, Any], 
                         expected_output: List[List[int]]) -> bool:
        """Validate solution against expected output"""
        if 'output_grid' not in solution:
            return False
        return (solution['output_grid'] == expected_output).all()
        
    def _learn_from_execution(self, task: Dict[str, Any], 
                            solution: Dict[str, Any], 
                            success: bool):
        """Learn from execution results"""
        self.strategy_selector.update_from_execution(task, solution, success)
        self.concept_extractor.update_from_execution(task, solution, success)
        self.pattern_graph.update_from_execution(task, solution, success)
        
    def _update_performance(self, solution: Dict[str, Any]):
        """Update performance history"""
        self.performance_history.append({
            'task_id': self.current_task.get('task_id', 'unknown'),
            'success': solution.get('success', False),
            'confidence': solution.get('confidence', 0.0),
            'time_taken': solution.get('time_taken', 0.0)
        })
        
    def _get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if not self.performance_history:
            return {}
            
        return {
            'success_rate': sum(1 for p in self.performance_history if p['success']) / len(self.performance_history),
            'avg_confidence': sum(p['confidence'] for p in self.performance_history) / len(self.performance_history),
            'avg_time': sum(p['time_taken'] for p in self.performance_history) / len(self.performance_history)
        }
