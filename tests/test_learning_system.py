import unittest
import json
import numpy as np
from pathlib import Path
from src.learning.learning_orchestrator import LearningOrchestrator
from src.llm.llm_interface import LLMInterface
from src.dsl.primitives import DynamicPrimitiveLibrary, DSLPrimitive
import os
from dotenv import load_dotenv
from src.strategy.meta_strategy import MetaStrategyEngine
from src.learning.concept_formation import ConceptFormationEngine
from src.utils.json_validator import JSONValidator

class TestLearningSystem(unittest.TestCase):
    """Test cases for the learning system"""

    def setUp(self):
        """Set up test fixtures"""
        self.llm = MockLLMInterface()
        self.meta_strategy = MetaStrategyEngine(self.llm)
        self.concept_formation = ConceptFormationEngine(self.llm)
        
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Copy .env.example to .env and add your API key.")
            
        self.llm = LLMInterface(api_key)
        self.orchestrator = LearningOrchestrator(self.llm)
        
        # Load example task
        task_path = Path(__file__).parent.parent / 'data' / 'example_task.json'
        with open(task_path) as f:
            self.example_task = json.load(f)
            
    def test_concept_learning(self):
        """Test that system can learn concepts from examples"""
        # Learn from first training example
        result = self.orchestrator.learn_from_task(
            self.example_task['train'][0]
        )
        
        # Verify concepts were learned
        self.assertTrue(len(result['learned_items']) > 0)
        
        # Check if concepts are valid
        for item in result['learned_items']:
            if item['type'] == 'concept':
                concept = item['item']
                self.assertIn('id', concept)
                self.assertIn('description', concept)
                self.assertIn('rules', concept)
                
    def test_primitive_learning(self):
        """Test that system can discover new primitives"""
        # Create a mock solution that inverts values and adds border
        solution = {
            'success_rate': 1.0,
            'steps': [
                {'primitive': 'invert', 'params': {}},
                {'primitive': 'replace_border', 'params': {'old_val': 0, 'new_val': 1}}
            ]
        }
        
        result = self.orchestrator.learn_from_task(
            self.example_task['train'][0],
            solution
        )
        
        # Verify primitive was discovered
        primitives_discovered = [
            item for item in result['learned_items']
            if item['type'] == 'primitive'
        ]
        self.assertTrue(len(primitives_discovered) > 0)
        
        # Check primitive properties
        for item in primitives_discovered:
            primitive = item['item']
            self.assertIn('id', primitive)
            self.assertIn('name', primitive)
            self.assertIn('description', primitive)
            self.assertIn('parameters', primitive)
            
    def test_continuous_learning(self):
        """Test that system improves with more examples"""
        print("\n=== Testing Continuous Learning ===")
        initial_success = None
        
        # Learn from first example
        solution1 = {
            'success_rate': 0.8,
            'strategy_id': 'test_strategy',
            'steps': [
                {'primitive': 'invert', 'params': {}},
                {'primitive': 'replace_border', 'params': {'old_val': 0, 'new_val': 1}}
            ]
        }
        
        print("\nLearning from first example...")
        learn_result1 = self.orchestrator.learn_from_task(
            self.example_task['train'][0],
            solution1
        )
        print(f"Learned items from first example: {json.dumps(learn_result1.get('learned_items', []), indent=2)}")
        
        # Try solving second example
        print("\nSolving second example...")
        result1 = self.orchestrator.solve_task(
            self.example_task['train'][1]
        )
        initial_success = result1.get('success_rate', 0.0)
        print(f"Initial solution steps: {json.dumps(result1.get('steps', []), indent=2)}")
        print(f"Initial success rate: {initial_success}")
        
        # Learn from second example
        solution2 = {
            'success_rate': 0.9,
            'strategy_id': 'test_strategy',
            'steps': [
                {'primitive': 'invert', 'params': {}},
                {'primitive': 'get_border', 'params': {}},
                {'primitive': 'replace_border', 'params': {'old_val': 0, 'new_val': 1}}
            ]
        }
        
        print("\nLearning from second example...")
        learn_result2 = self.orchestrator.learn_from_task(
            self.example_task['train'][1],
            solution2
        )
        print(f"Learned items from second example: {json.dumps(learn_result2.get('learned_items', []), indent=2)}")
        
        # Try solving test example
        print("\nSolving test example...")
        result2 = self.orchestrator.solve_task(
            self.example_task['test'][0]
        )
        print(f"Final solution steps: {json.dumps(result2.get('steps', []), indent=2)}")
        print(f"Final success rate: {result2.get('success_rate', 0.0)}")
        
        # System should show improved success rate after learning from more examples
        self.assertGreater(result2.get('success_rate', 0.0), initial_success)
        
    def test_feedback_incorporation(self):
        """Test that system can incorporate feedback"""
        print("\n=== Testing Feedback Incorporation ===")
        
        # Initial learning
        print("\nInitial learning...")
        result1 = self.orchestrator.learn_from_task(
            self.example_task['train'][0]
        )
        print(f"Initial learned items: {json.dumps(result1.get('learned_items', []), indent=2)}")
        
        # Get concept that needs feedback
        concept_id = None
        for item in result1['learned_items']:
            if item['type'] == 'concept':
                concept_id = item['item']['id']
                print(f"\nFound concept to give feedback to: {item['item']['id']}")
                print(f"Current concept description: {item['item']['description']}")
                break
                
        self.assertIsNotNone(concept_id)
        
        # Provide feedback
        feedback = {
            'type': 'concept',
            'concept_id': concept_id,
            'feedback': 'The concept should focus more on border patterns',
            'suggestions': ['Consider border cell relationships']
        }
        
        print("\nProviding feedback...")
        print(f"Feedback: {json.dumps(feedback, indent=2)}")
        self.orchestrator.incorporate_feedback(feedback)
        
        # Learn again with same example
        print("\nLearning again after feedback...")
        result2 = self.orchestrator.learn_from_task(
            self.example_task['train'][0]
        )
        print(f"New learned items: {json.dumps(result2.get('learned_items', []), indent=2)}")
        
        # Verify concepts evolved
        evolved_concepts = [
            item for item in result2['learned_items']
            if item['type'] == 'concept' and 'border' in item['item']['description'].lower()
        ]
        print(f"\nFound {len(evolved_concepts)} evolved concepts with border focus")
        for concept in evolved_concepts:
            print(f"Evolved concept: {concept['item']['description']}")
            
        self.assertTrue(len(evolved_concepts) > 0)
        
    def test_strategy_adaptation(self):
        """Test that system can adapt strategies based on performance"""
        # Initial strategy
        context = self.orchestrator._extract_context(
            self.example_task['train'][0]
        )
        
        strategy1 = self.orchestrator.strategy_engine.select_strategy(context)
        
        # Update with poor performance
        self.orchestrator.strategy_engine.update_performance(
            strategy1['id'],
            context,
            0.3,  # Poor performance
            {'error': 'Strategy ineffective for this task'}
        )
        
        # Get new strategy
        strategy2 = self.orchestrator.strategy_engine.select_strategy(context)
        
        # Verify strategy evolved meaningfully
        self.assertNotEqual(strategy1['id'], strategy2['id'], "Strategy should have a new ID after adaptation")
        
        # Check that the strategy has actually changed in a meaningful way
        self.assertNotEqual(
            strategy1.get('steps', []),
            strategy2.get('steps', []),
            "New strategy should have different steps than the original"
        )
        
        # Check that the strategy description reflects learning from failure
        self.assertIn(
            'confidence',
            strategy2,
            "Adapted strategy should include a confidence score"
        )
        
        # Verify the strategy has a meaningful description
        self.assertTrue(
            len(strategy2.get('description', '')) > 20,
            "Strategy should have a meaningful description explaining its approach"
        )
        
    def test_invalid_input_handling(self):
        """Test system's response to invalid inputs"""
        # TODO: Implement tests for:
        # - Empty input
        # - Malformed JSON
        # - Invalid grid dimensions
        # - Missing required fields
        pass

    def test_memory_management(self):
        """Test system's memory usage and cleanup"""
        # TODO: Implement tests for:
        # - Memory usage during learning
        # - Cleanup after task completion
        # - Handling of large datasets
        # - Resource allocation and deallocation
        pass

    def test_end_to_end_workflow(self):
        """Test complete workflow from learning to application"""
        # TODO: Implement tests for:
        # - Full pipeline execution
        # - Component interaction
        # - State management
        # - Cross-component data flow
        pass

    def test_concept_generalization(self):
        """Test how well learned concepts generalize"""
        # TODO: Implement tests for:
        # - Performance on unseen examples
        # - Overfitting detection
        # - Concept stability
        # - Pattern recognition accuracy
        pass

    def test_state_persistence(self):
        """Test saving and loading system state"""
        # TODO: Implement tests for:
        # - Concept persistence
        # - Strategy persistence
        # - Recovery from interruption
        # - State versioning
        pass

    def test_system_resilience(self):
        """Test system's ability to handle adverse conditions"""
        # TODO: Implement tests for:
        # - Conflicting patterns
        # - Noisy data
        # - Resource constraints
        # - Error recovery
        pass

    def test_performance_benchmarks(self):
        """Test system performance metrics"""
        # TODO: Implement tests for:
        # - Processing time benchmarks
        # - Memory usage benchmarks
        # - Scaling with input size
        # - Resource utilization
        pass

    def test_concurrent_operations(self):
        """Test system behavior under concurrent operations"""
        # TODO: Implement tests for:
        # - Parallel task processing
        # - Resource sharing
        # - State consistency
        # - Race condition handling
        pass

    def test_regression_suite(self):
        """Test suite for known issues and edge cases"""
        # TODO: Implement tests for:
        # - Previously identified bugs
        # - Edge cases
        # - Corner cases
        # - Boundary conditions
        pass

class MockLLMInterface:
    """Mock LLM interface for testing"""
    def __init__(self):
        self.response_count = 0
        self.poor_performing_strategies = set()
        
    def get_completion(self, prompt: str, schema=None) -> str:
        """Return mock responses for testing"""
        self.response_count += 1
        
        if schema == JSONValidator.STRATEGY_SCHEMA:
            # Check if we're generating a strategy after poor performance
            if "Poor Strategies:" in prompt and "[]" not in prompt:
                # Return an improved strategy with different steps
                return json.dumps({
                    "strategy": {
                        "id": f"strategy_{self.response_count}",
                        "name": "Improved Test Strategy",
                        "description": "An improved strategy that addresses previous failures",
                        "steps": [{"primitive": "improved_primitive", "params": {"threshold": 0.8}}],
                        "applicability": "Testing with improvements",
                        "confidence": 0.9
                    }
                })
            else:
                # Return initial strategy
                return json.dumps({
                    "strategy": {
                        "id": f"strategy_{self.response_count}",
                        "name": "Test Strategy",
                        "description": "A test strategy for unit testing",
                        "steps": [{"primitive": "test_primitive", "params": {}}],
                        "applicability": "Testing only",
                        "confidence": 0.8
                    }
                })
        elif schema == JSONValidator.CONCEPT_SCHEMA:
            return json.dumps({
                "concepts": [{
                    "id": "test_concept",
                    "name": "Border Pattern Concept",
                    "description": "A concept focused on border patterns and transformations",
                    "rules": ["Analyze border cells", "Track border changes"],
                    "confidence": 0.8
                }]
            })
        elif schema == JSONValidator.PRIMITIVE_SCHEMA:
            return json.dumps({
                "primitive": {
                    "id": "test_primitive",
                    "name": "Test Primitive",
                    "description": "A test primitive for border pattern analysis",
                    "parameters": {"param1": "int"},
                    "implementation_guide": "Mock implementation",
                    "applicability": "Testing only",
                    "examples": ["test example"]
                }
            })
        else:
            # Default mock response
            return json.dumps({"response": "Mock response for testing"})

if __name__ == '__main__':
    unittest.main()
