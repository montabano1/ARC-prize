import os
import json
import numpy as np
import logging
import asyncio
from dotenv import load_dotenv
from src.llm.llm_interface import LLMInterface
from src.learning.learning_orchestrator import LearningOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Added timestamp, module name and level
)
logger = logging.getLogger(__name__)

# Disable debug logging for other modules
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)

def display_grid(grid):
    """Display a grid in a readable format."""
    if isinstance(grid, list):
        grid = np.array(grid)
    print(grid)

async def main():
    """Main function to run the learning system"""
    
    # Load environment variables
    print("Loading environment variables...")
    load_dotenv()
    
    # Get API key
    print("Getting API key...")
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("No OpenAI API key found in environment variables")
    print("API key loaded successfully")
    
    # Initialize LLM interface
    print("Initializing LLM interface...")
    llm = LLMInterface(api_key)
    print("LLM interface initialized")
    
    # Initialize learning system
    print("Initializing learning system...")
    learning_orchestrator = LearningOrchestrator(llm)
    print("Learning system initialized")
    
    # Load example task
    print("Loading example task...")
    task_data = {
        'train': [
            {
                'input': [[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]],
                'output': [[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]]
            },
            {
                'input': [[0, 0, 0, 0],
                         [0, 1, 1, 0],
                         [0, 1, 1, 0],
                         [0, 0, 0, 0]],
                'output': [[1, 1, 1, 1],
                          [1, 0, 0, 1],
                          [1, 0, 0, 1],
                          [1, 1, 1, 1]]
            }
        ],
        'test': [
            {
                'input': [[0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0]],
                'output': [[1, 1, 1, 1, 1],
                          [1, 0, 0, 0, 1],
                          [1, 0, 0, 0, 1],
                          [1, 0, 0, 0, 1],
                          [1, 1, 1, 1, 1]]
            }
        ]
    }
    print("Example task loaded")
    
    print("Starting to learn from task...")
    # Learn from task
    await learning_orchestrator.learn_from_task(task_data)
    print("Finished learning from task")
    
    # Print training examples
    print("\nTraining examples:\n")
    for i, example in enumerate(task_data['train'], 1):
        print(f"Example {i}:")
        print("Input:")
        print(np.array(example['input']))
        print("\nOutput:")
        print(np.array(example['output']))
        print()
    
    print("Learning unified strategy from all examples...\n")
    
    # Print test examples and solve
    print("\nTesting examples:\n")
    for i, test in enumerate(task_data['test'], 1):
        print(f"Test {i}:")
        print("Input:")
        print(np.array(test['input']))
        print("\nExpected Output:")
        print(np.array(test['output']))
        print()
        
        # Try to solve the test case
        solution = await learning_orchestrator.solve_task(test)
        if solution and 'output' in solution:
            print("\nGenerated Output:")
            print(np.array(solution['output']))
            
            # Calculate accuracy
            if isinstance(solution['output'], list):
                actual = np.array(solution['output'])
                expected = np.array(test['output'])
                accuracy = np.mean(actual == expected) * 100
                print(f"\nAccuracy: {accuracy:.2f}%")
            else:
                print("\nCould not calculate accuracy - output is not a list")
        else:
            print("\nNo solution generated")
            
if __name__ == "__main__":
    asyncio.run(main())
