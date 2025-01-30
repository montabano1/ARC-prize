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
        
    # Initialize components with LLM integration
    llm = LLMInterface(api_key)
    primitive_library = DynamicPrimitiveLibrary(llm)
    concept_formation = ConceptFormation()
    curriculum = Curriculum()
    performance_optimizer = PerformanceOptimizer(llm, primitive_library)
    dsl_synthesizer = DSLSynthesizer(primitive_library, llm)
    task_analyzer = TaskAnalyzer(llm, concept_formation)
    
    task_manager = TaskManager(
        task_analyzer=task_analyzer,
        dsl_synthesizer=dsl_synthesizer,
        concept_formation=concept_formation,
        curriculum=curriculum,
        performance_optimizer=performance_optimizer
    )
    
    while True:
        print("\n=== ARC Task Analysis ===")
        print("1) Analyze task")
        print("2) View curriculum")
        print("3) View concepts")
        print("4) View primitives")
        print("5) View learning history")
        print("6) Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == '6':
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
                task_features = task_analyzer.analyze_task(task_data)
                
                # Display results
                print("\n" + "═" * 50)
                print("Task Analysis Results")
                print("═" * 50)
                
                print("\nTask Properties:")
                print("─" * 33)
                print(f"Grid Size: {task_features.grid_size}")
                print(f"Unique Colors: {task_features.unique_colors}")
                print(f"Complexity: {task_features.pattern_complexity:.2f}")
                print(f"Difficulty: {task_features.overall_difficulty:.2f}")
                
                print("\nIdentified Patterns:")
                print("─" * 33)
                print(f"\nObject:\n• {task_features.identified_patterns.get('object', '')}")
                print(f"\nTransformation:\n• {task_features.identified_patterns.get('transformation', '')}")
                print(f"\nRelationship:\n• {task_features.identified_patterns.get('relationship', '')}")
                print(f"\nAbstract:\n• {task_features.identified_patterns.get('abstract', '')}")
                
                print("\nUnified Strategy:")
                print("─" * 33)
                strategy = task_features.unified_strategy or "No unified strategy identified yet"
                print(strategy)
                
                print("\nConfidence Scores:")
                print("─" * 33)
                for pattern_type, score in task_features.confidence_scores.items():
                    print(f"{pattern_type:<12}: {score:.2f}")
                
                print("\nExtracted Concepts:")
                print("─" * 33)
                for concept in task_features.extracted_concepts:
                    print(f"\n{concept.name}:")
                    print(f"• Description: {concept.description}")
                
                print("\nGenerating DSL Program...")
                print("─" * 33)
                
                # Generate and optimize program
                program = dsl_synthesizer.synthesize_program(task_features)
                if program:
                    # Evaluate initial performance
                    success_rate = performance_optimizer._evaluate_program(program, task_data)
                    print(f"\nInitial Success Rate: {success_rate:.2f}")
                    
                    # Optimize if needed
                    if success_rate < 0.9:
                        print("\nOptimizing program...")
                        program = performance_optimizer.optimize_program(
                            program, task_data, task_features, success_rate
                        )
                        new_success = performance_optimizer._evaluate_program(program, task_data)
                        print(f"Optimized Success Rate: {new_success:.2f}")
                        
                        if new_success > success_rate:
                            print("\nOptimization improved performance!")
                            if primitive_library.primitive_history:
                                print("New primitives learned:", len(primitive_library.primitive_history))
                else:
                    print("Could not generate a DSL program for this task")
                
            except Exception as e:
                print(f"Error processing task: {e}")
                
        elif choice == '2':
            print("\nCurriculum Status:")
            print("─" * 33)
            # Add curriculum viewing logic
            
        elif choice == '3':
            print("\nLearned Concepts:")
            print("─" * 33)
            # Add concept viewing logic
            
        elif choice == '4':
            print("\nAvailable Primitives:")
            print("─" * 33)
            for name, primitive in primitive_library.primitives.items():
                print(f"\n{name}:")
                print(f"• Description: {primitive.description}")
                print(f"• Success Rate: {primitive.success_rate:.2f}")
                print(f"• Usage Count: {primitive.usage_count}")
                
        elif choice == '5':
            print("\nLearning History:")
            print("─" * 33)
            print("\nPrimitive Learning History:")
            for entry in primitive_library.primitive_history:
                print(f"\nTimestamp: {entry['timestamp']}")
                print(f"Primitive: {entry['primitive'].name}")
                print(f"Success Rate: {entry['performance_data']['success_rate']:.2f}")
                
            print("\nOptimization History:")
            for entry in performance_optimizer.optimization_history:
                print(f"\nOriginal Success: {entry['original_success']:.2f}")
                print(f"Optimized Success: {entry['optimized_success']:.2f}")
                if 'feedback' in entry:
                    print("Improvements:", entry['feedback'].get('failure_analysis', ''))

if __name__ == "__main__":
    main()
