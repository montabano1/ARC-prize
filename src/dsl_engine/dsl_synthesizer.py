from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class DSLPrimitive:
    name: str
    operation: callable
    input_types: List[str]
    output_type: str
    complexity: float

@dataclass
class DSLProgram:
    primitives: List[DSLPrimitive]
    program_str: str
    confidence: float
    execution_time: float

class DSLSynthesizer:
    def __init__(self):
        self.primitives = {}
        self.programs = {}
        self.max_program_length = 10
        
    def register_primitive(self, primitive: DSLPrimitive):
        """Register a new DSL primitive operation."""
        self.primitives[primitive.name] = primitive
        
    def synthesize_program(self, input_output_pairs: List[Tuple[np.ndarray, np.ndarray]], 
                         max_attempts: int = 1000) -> Optional[DSLProgram]:
        """Synthesize a DSL program that satisfies the input-output pairs."""
        best_program = None
        best_score = float('-inf')
        
        for _ in range(max_attempts):
            program = self._generate_candidate_program()
            if program is None:
                continue
                
            score = self._evaluate_program(program, input_output_pairs)
            if score > best_score:
                best_score = score
                best_program = program
                
            if best_score == 1.0:  # Perfect match
                break
                
        return best_program
        
    def _generate_candidate_program(self) -> Optional[DSLProgram]:
        """Generate a candidate DSL program."""
        program_length = np.random.randint(1, self.max_program_length + 1)
        primitives = []
        program_str = ""
        
        try:
            current_type = "grid"  # Starting type for ARC tasks
            for _ in range(program_length):
                # Filter primitives that can accept current_type
                valid_primitives = [
                    p for p in self.primitives.values()
                    if p.input_types[0] == current_type
                ]
                
                if not valid_primitives:
                    break
                    
                chosen_primitive = np.random.choice(valid_primitives)
                primitives.append(chosen_primitive)
                program_str += f"{chosen_primitive.name}("
                current_type = chosen_primitive.output_type
                
            program_str = self._format_program_string(program_str)
            return DSLProgram(
                primitives=primitives,
                program_str=program_str,
                confidence=0.0,
                execution_time=0.0
            )
            
        except Exception:
            return None
            
    def _evaluate_program(self, program: DSLProgram, 
                         input_output_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Evaluate how well a program matches the input-output pairs."""
        try:
            correct = 0
            total = len(input_output_pairs)
            
            for input_grid, expected_output in input_output_pairs:
                actual_output = self._execute_program(program, input_grid)
                if np.array_equal(actual_output, expected_output):
                    correct += 1
                    
            return correct / total
            
        except Exception:
            return 0.0
            
    def _execute_program(self, program: DSLProgram, input_grid: np.ndarray) -> np.ndarray:
        """Execute a DSL program on an input grid."""
        current_output = input_grid
        for primitive in program.primitives:
            current_output = primitive.operation(current_output)
        return current_output
        
    def _format_program_string(self, program_str: str) -> str:
        """Format the program string to be more readable."""
        # Add proper closing parentheses
        program_str += ")" * program_str.count("(")
        return program_str
