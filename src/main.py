import os
import json
from dotenv import load_dotenv
import numpy as np

from src.llm.llm_interface import LLMInterface
from src.task_assessment.task_analyzer import TaskAnalyzer
from src.dsl.primitives import DynamicPrimitiveLibrary
from src.dsl.synthesizer import DSLSynthesizer
from src.concept.concept_formation import ConceptFormation
from src.curriculum.curriculum import Curriculum
from src.learning.performance_optimizer import PerformanceOptimizer

def display_grid(grid):
    """Display a grid in a readable format."""
    if isinstance(grid, list):
        grid = np.array(grid)
    for row in grid:
        print(' '.join(str(cell) for cell in row))

class TaskManager:
    def __init__(self, task_analyzer: TaskAnalyzer, dsl_synthesizer: DSLSynthesizer,
                 concept_formation: ConceptFormation, curriculum: Curriculum,
                 performance_optimizer: PerformanceOptimizer):
        self.task_analyzer = task_analyzer
        self.dsl_synthesizer = dsl_synthesizer
        self.concept_formation = concept_formation
        self.curriculum = curriculum
        self.performance_optimizer = performance_optimizer
        
    def process_task(self, task: dict) -> dict:
        """Process an ARC task following the flow diagram."""
        try:
            if not task.get('train'):
                raise KeyError("'train' data is missing or empty")
            
            if not isinstance(task['train'], list) or not task['train']:
                raise ValueError("'train' data must be a non-empty list")
                
            first_example = task['train'][0]
            
            if not isinstance(first_example, dict):
                raise ValueError("Training example must be a dictionary")
                
            # Step 1: Task Assessment
            task_features = self.task_analyzer.analyze_task(task)
            return task_features
            
        except Exception as e:
            raise

def main():
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
        
    # Initialize components
    llm = LLMInterface(api_key)
    primitive_library = DynamicPrimitiveLibrary()
    concept_formation = ConceptFormation()
    curriculum = Curriculum()
    performance_optimizer = PerformanceOptimizer()
    dsl_synthesizer = DSLSynthesizer(primitive_library)
    task_analyzer = TaskAnalyzer(llm, concept_formation)
    
    while True:
        print("\n=== ARC Task Analysis ===")
        print("1) Analyze task")
        print("2) View curriculum")
        print("3) View concepts")
        print("4) View primitives")
        print("5) Exit")
        
        choice = input("\nEnter your choice (1-5): ")
        
        if choice == '5':
            break
            
        elif choice == '1':
            # Get task file
            task_path = input("\nTask file (default: data/example_task.json): ")
            if not task_path:
                task_path = "data/example_task.json"
                
            try:
                # Load task
                with open(task_path) as f:
                    task_data = json.load(f)
                    
                print("\nAnalyzing task patterns...")
                
                # Analyze task
                result = task_analyzer.analyze_task(task_data)
                
                # Display results
                print("\n" + "═" * 50)
                print("Task Analysis Results")
                print("═" * 50)
                
                # Task properties section
                print("\nTask Properties:")
                print("─" * 25)
                print(f"Grid Size: {result.grid_size}")
                print(f"Unique Colors: {result.unique_colors}")
                print(f"Complexity: {result.pattern_complexity:.2f}")
                print(f"Difficulty: {result.overall_difficulty:.2f}")
                
                # Patterns section
                print("\nIdentified Patterns:")
                print("─" * 25)
                if result.identified_patterns:
                    for category, patterns in result.identified_patterns.items():
                        if patterns:  # Only show categories with patterns
                            has_non_unified = any(not p.get('is_unified_strategy', False) for p in patterns)
                            if has_non_unified:
                                print(f"\n{category.title()}:")
                                for pattern in patterns:
                                    if pattern.get('is_unified_strategy'):
                                        continue
                                    desc = pattern.get('description', '').rstrip('.')
                                    print(f"• {desc}")
                    
                    # Show unified strategy at the end
                    unified_found = False
                    for pattern in result.identified_patterns.get('abstract', []):
                        if pattern.get('is_unified_strategy'):
                            print("\nUnified Strategy:")
                            print("─" * 25)
                            desc = pattern.get('description', 'No unified strategy found').rstrip('.')
                            print(f"• {desc}")
                            unified_found = True
                            break
                else:
                    print("No patterns identified")
                    
                # Confidence scores section
                print("\nConfidence Scores:")
                print("─" * 25)
                if result.confidence_scores:
                    max_category_len = max(len(category) for category in result.confidence_scores.keys())
                    for category, score in result.confidence_scores.items():
                        category_padded = category.title().ljust(max_category_len)
                        score_str = f"{score:.2f}"
                        print(f"{category_padded}: {score_str}")
                else:
                    print("No confidence scores available")
                    
                # Show extracted concepts
                print("\nExtracted Concepts:")
                print("─" * 25)
                if result.extracted_concepts:
                    for concept in result.extracted_concepts:
                        print(f"\n{concept.name.title()}:")
                        print(f"• Description: {concept.description}")
                        if concept.dsl_template:
                            print("• DSL Template:")
                            for step in concept.dsl_template:
                                params = ", ".join(f"{k}={v}" for k, v in step.get('params', {}).items())
                                print(f"  - {step['primitive']}({params})")
                else:
                    print("No concepts extracted")
                    
                # Generate and test DSL program
                print("\nGenerating DSL Program...")
                print("─" * 25)
                program = dsl_synthesizer.synthesize_program(result)
                
                if program:
                    print(program.description)
                    
                    # Test program on training examples
                    print("\nTesting Program:")
                    print("─" * 25)
                    success = 0
                    total = 0
                    
                    for example in task_data['train']:
                        input_grid = np.array(example['input'])
                        expected = np.array(example['output'])
                        actual = dsl_synthesizer.execute_program(program, input_grid)
                        
                        matches = np.array_equal(actual, expected)
                        total += 1
                        if matches:
                            success += 1
                        
                        print(f"Example {total}: {'✓' if matches else '✗'}")
                    
                    success_rate = success / total if total > 0 else 0
                    print(f"\nSuccess Rate: {success_rate:.0%}")
                    
                    # Update curriculum based on performance
                    curriculum.update_progress(success_rate)
                else:
                    print("Could not generate a DSL program for this task")
                    
                print("\n" + "═" * 50)
                
            except FileNotFoundError:
                print(f"Error: File {task_path} not found")
            except json.JSONDecodeError:
                print(f"Error: File {task_path} is not valid JSON")
            except Exception as e:
                print(f"Error processing task: {str(e)}")
                import traceback
                print("\nFull error traceback:")
                traceback.print_exc()
                
        elif choice == '2':
            # View curriculum status
            print("\nCurriculum Status:")
            print("─" * 25)
            print(f"Current Level: {curriculum.current_level:.2f}")
            print("\nFocus Areas:")
            for area in curriculum.focus_areas:
                print(f"• {area}")
                
        elif choice == '3':
            # View learned concepts
            print("\nLearned Concepts:")
            print("─" * 25)
            if concept_formation.concepts:
                for name, concept in concept_formation.concepts.items():
                    print(f"\n{name.title()}:")
                    print(f"• Description: {concept.description}")
                    print(f"• Usage Count: {concept.usage_count}")
                    print(f"• Success Rate: {concept.success_rate:.0%}")
            else:
                print("No concepts learned yet")
                
        elif choice == '4':
            # View available primitives
            print("\nAvailable DSL Primitives:")
            print("─" * 25)
            for name in primitive_library.list_primitives():
                primitive = primitive_library.get_primitive(name)
                params = ", ".join(primitive.parameters)
                print(f"\n{name}({params}):")
                print(f"• {primitive.description}")
                print(f"• Complexity: {primitive.complexity:.2f}")
                print(f"• Usage Count: {primitive.usage_count}")
                print(f"• Success Rate: {primitive.success_rate:.0%}")
                
if __name__ == "__main__":
    main()
