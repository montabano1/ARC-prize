import sys
import traceback
import numpy as np
from dotenv import load_dotenv
import os
import asyncio
import json
import glob

def handle_error(e: Exception):
    """Handle error by printing full info and exiting"""
    print("\nError occurred:", file=sys.stderr)
    print(f"Error type: {type(e).__name__}", file=sys.stderr)
    print(f"Error message: {str(e)}", file=sys.stderr)
    print("\nFull traceback:", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

def display_grid(grid):
    """Display a grid in a readable format."""
    if isinstance(grid, list):
        grid = np.array(grid)
    print(grid)

def load_puzzle(puzzle_path: str = None):
    """Load a puzzle from file or use default example"""
    if puzzle_path:
        with open(puzzle_path, 'r') as f:
            return json.load(f)
    
    # Return default example if no path provided
    return {
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

def list_available_puzzles(puzzle_dir: str):
    """List all available puzzle files"""
    puzzle_files = glob.glob(os.path.join(puzzle_dir, "*.json"))
    if not puzzle_files:
        print("No puzzle files found in", puzzle_dir)
        return []
    
    print("\nAvailable puzzles:")
    for i, path in enumerate(puzzle_files, 1):
        name = os.path.basename(path)
        print(f"{i}. {name}")
    return puzzle_files

async def main():
    """Main function to run the learning system"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("No OpenAI API key found in environment variables")
        
        # Initialize components
        from src.llm.llm_interface import LLMInterface
        from src.learning.learning_orchestrator import LearningOrchestrator
        from src.dsl.synthesizer import DSLProgram, DSLPrimitive
        
        llm = LLMInterface()
        learning_orchestrator = LearningOrchestrator()
        
        # Check for puzzles directory
        puzzle_dir = os.path.join(os.path.dirname(__file__), "..", "puzzles")
        if not os.path.exists(puzzle_dir):
            os.makedirs(puzzle_dir)
            print("\nCreated puzzles directory. Please add your puzzle JSON files there.")
            print("Using default example puzzle for now...")
            task_data = load_puzzle()
        else:
            # List available puzzles
            puzzle_files = list_available_puzzles(puzzle_dir)
            
            if puzzle_files:
                try:
                    choice = input("\nEnter puzzle number to solve (or press Enter for default example): ")
                except (EOFError, KeyboardInterrupt):
                    print("\nUsing default example...")
                    task_data = load_puzzle()
                else:
                    if not choice:
                        print("\nUsing default example puzzle...")
                        task_data = load_puzzle()
                    else:
                        try:
                            idx = int(choice) - 1
                            if 0 <= idx < len(puzzle_files):
                                print(f"\nLoading puzzle: {os.path.basename(puzzle_files[idx])}")
                                task_data = load_puzzle(puzzle_files[idx])
                            else:
                                print("Invalid choice. Using default example...")
                                task_data = load_puzzle()
                        except ValueError:
                            print("Invalid input. Using default example...")
                            task_data = load_puzzle()
            else:
                print("\nUsing default example puzzle...")
                task_data = load_puzzle()
        
        # Display training examples
        print("\nAnalyzing training examples...")
        for i, example in enumerate(task_data['train'], 1):
            print(f"\nTraining Example {i}:")
            print("Input:")
            display_grid(example['input'])
            print("\nOutput:")
            display_grid(example['output'])
        
        # Learn from task
        result = await learning_orchestrator.learn_from_task(task_data)
        
        # Get next task based on performance
        accuracy = result.get('accuracy', 0.0)
        next_task = await learning_orchestrator.curriculum_manager.select_next_task(accuracy)
        
        # Print results
        if result.get('patterns'):
            print("\nIdentified Patterns:")
            for pattern in result['patterns']:
                print(f"- {pattern.get('description', 'No description available')}")
                
        if result.get('strategy'):
            print("\nGenerated Strategy:")
            print(result['strategy'].get('description', 'No strategy description available'))
            
        if result.get('progress'):
            print("\nLearning Progress:")
            print(f"Overall Progress: {result['progress'].get('overall_progress', 0.0):.2%}")
            
        if next_task:
            print(f"\nNext recommended task: {next_task}")
        else:
            print("\nNo next task recommended.")
        
        # Process test example
        print("\nTest Example:")
        print("Input:")
        display_grid(task_data['test'][0]['input'])
        print("\nExpected Output:")
        display_grid(task_data['test'][0]['output'])
        
        # Get explanation from LLM
        print("\nChatGPT: Analyzing the pattern...")
        explanation = await llm.explain_strategy({
            'task': task_data,
            'context': {'task_type': 'grid_transformation'}
        })
        if explanation:
            print("\nI've identified the pattern. In all examples, we need to:")
            print(explanation)
        else:
            print("\nI apologize, but I'm having trouble explaining this pattern clearly.")
        
        # Generate and validate hypothesis
        print("\nChatGPT: Generating hypothesis...")
        hypothesis = await learning_orchestrator.hypothesis_former.form_hypothesis(task_data['train'])
        if hypothesis:
            print("\nHypothesis generated successfully!")
            print(hypothesis.get('description', 'No description available'))
            
            # Validate hypothesis
            print("\nValidating hypothesis...")
            validation = await learning_orchestrator.pattern_validator.validate_pattern(
                hypothesis,
                task_data['train']
            )
            if validation.get('is_valid'):
                print("\nThe hypothesis has been validated! The pattern is consistent across all examples.")
            else:
                print("\nThe hypothesis needs refinement. Some inconsistencies were found.")
        else:
            print("\nI was unable to form a clear hypothesis for this pattern.")
        
        # Check for knowledge gaps
        gaps = await learning_orchestrator.curriculum_manager.identify_knowledge_gaps()
        if gaps:
            print("\nChatGPT: I've identified some areas where we need more practice:", ", ".join(gaps))
        
        # Ask if user wants to try another puzzle
        choice = input("\nWould you like to try another puzzle? (y/n): ")
        if choice.lower() == 'y':
            await main()
        
    except Exception as e:
        handle_error(e)

if __name__ == "__main__":
    asyncio.run(main())
