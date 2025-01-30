import numpy as np
from src.integration.solution_synthesis import SolutionComponent, Solution, StrategyExecutor

# Mock task
task = {
    'train': [{
        'input': [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ],
        'output': [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
    }]
}

# Mock components
components = [
    SolutionComponent(
        id="border",
        type="concept",
        content={
            'description': "The border transformation: Set all non-center points to 1"
        },
        dependencies=[],
        confidence=1.0
    ),
    SolutionComponent(
        id="center",
        type="concept",
        content={
            'description': "The central point transformation: Set center to 0"
        },
        dependencies=[],
        confidence=1.0
    )
]

# Create solution
solution = Solution(
    components=components,
    execution_order=["border", "center"],
    validation_results={},
    performance_metrics={},
    confidence=1.0
)

# Create executor (without dependencies)
class MockStrategySelector:
    def __init__(self):
        pass
    def get_strategy(self, strategy_id):
        return None

class MockLLM:
    def __init__(self):
        pass
    def analyze_pattern(self, *args, **kwargs):
        return None

executor = StrategyExecutor(strategy_selector=MockStrategySelector(), llm=MockLLM())

# Execute solution
print("Input grid:")
print(np.array(task['train'][0]['input']))

results = executor.execute_solution(solution, task)

print("\nExpected output:")
print(np.array(task['train'][0]['output']))

print("\nActual output:")
print(results['final_grid'])
