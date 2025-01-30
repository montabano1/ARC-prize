import click
import json
import yaml
from pathlib import Path
from typing import Dict, Any

from src.solver import ARCSolver

@click.group()
def cli():
    """ARC Prize Solver CLI"""
    pass

@cli.command()
@click.argument('task_file', type=click.Path(exists=True))
@click.option('--debug', is_flag=True, help='Enable debug mode')
def solve(task_file: str, debug: bool):
    """Solve a single ARC task"""
    solver = ARCSolver(debug=debug)
    
    with open(task_file, 'r') as f:
        task = json.load(f)
    
    solution = solver.solve(task)
    print(json.dumps(solution, indent=2))

@cli.command()
def interactive():
    """Start interactive session"""
    solver = ARCSolver()
    
    while True:
        command = input(">> ")
        if command == "exit":
            break
            
        parts = command.split()
        if not parts:
            continue
            
        if parts[0] == "load":
            task_id = parts[1]
            solver.load_task(task_id)
            print(f"Loaded task {task_id}")
            
        elif parts[0] == "show":
            if parts[1] == "current":
                state = solver.get_state()
                print(json.dumps(state, indent=2))
                
        elif parts[0] == "explain":
            if parts[1] == "solution":
                explanation = solver.explain_solution()
                print(explanation)
                
        elif parts[0] == "feedback":
            feedback = " ".join(parts[1:])
            solver.process_feedback(feedback)
            print("Feedback processed")
            
        elif parts[0] == "concepts":
            if parts[1] == "list":
                concepts = solver.list_concepts()
                print(json.dumps(concepts, indent=2))
                
        else:
            print(f"Unknown command: {command}")

@cli.command()
@click.argument('task_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True))
def train(task_dir: str):
    """Train on a directory of tasks"""
    solver = ARCSolver()
    
    task_files = Path(task_dir).glob("*.json")
    for task_file in task_files:
        with open(task_file, 'r') as f:
            task = json.load(f)
        solver.train(task)
        print(f"Trained on {task_file.name}")

if __name__ == "__main__":
    cli()
