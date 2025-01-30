import unittest
import json
import numpy as np
from pathlib import Path
from src.learning.learning_orchestrator import LearningOrchestrator
from src.llm.llm_interface import LLMInterface
from src.dsl.primitives import DynamicPrimitiveLibrary, DSLPrimitive
import os
from dotenv import load_dotenv

class TestLearningSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment variable
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Copy .env.example to .env and add your API key.")
            
        cls.llm = LLMInterface(api_key)
        cls.orchestrator = LearningOrchestrator(cls.llm)
        
        # Load example task
        task_path = Path(__file__).parent.parent / 'data' / 'example_task.json'
        with open(task_path) as f:
            cls.example_task = json.load(f)
            
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
        
if __name__ == '__main__':
    unittest.main()
