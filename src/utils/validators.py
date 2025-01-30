from typing import Dict, Any, List, Optional, Set
import json

class ValidationError(Exception):
    """Custom validation error"""
    pass

class SystemValidators:
    """Validators for system-wide invariants"""
    
    @staticmethod
    def validate_strategy_id_unique(strategy_id: str, existing_ids: Set[str]) -> bool:
        """
        Validate that a strategy ID is unique.
        Returns True if unique, False if not.
        """
        return strategy_id not in existing_ids

    @staticmethod
    def validate_learning_result(result: Dict[str, Any]) -> None:
        """
        Validate a learning result has at least one learned item.
        Raises ValidationError if invalid.
        """
        if not result.get('learned_items'):
            raise ValidationError("Learning result must contain at least one learned item")

    @staticmethod
    def validate_solution_steps(steps: List[Dict[str, Any]], available_primitives: Set[str]) -> None:
        """
        Validate that solution steps only use available primitives.
        Raises ValidationError if invalid.
        """
        for step in steps:
            if step['primitive'] not in available_primitives:
                raise ValidationError(f"Solution uses unknown primitive: {step['primitive']}")

    @staticmethod
    def validate_grid_dimensions(grid: List[List[int]]) -> None:
        """
        Validate that a grid has consistent dimensions.
        Raises ValidationError if invalid.
        """
        if not grid:
            raise ValidationError("Grid cannot be empty")
            
        width = len(grid[0])
        for row in grid:
            if len(row) != width:
                raise ValidationError("Grid rows must have consistent width")

    @staticmethod
    def validate_performance_score(score: float) -> None:
        """
        Validate that a performance score is between 0 and 1.
        Raises ValidationError if invalid.
        """
        if not 0 <= score <= 1:
            raise ValidationError("Performance score must be between 0 and 1")

    @staticmethod
    def validate_concept_confidence(confidence: float) -> None:
        """
        Validate that a concept confidence is between 0 and 1.
        Raises ValidationError if invalid.
        """
        if not 0 <= confidence <= 1:
            raise ValidationError("Concept confidence must be between 0 and 1")

    @staticmethod
    def validate_feedback_type(feedback_type: str) -> None:
        """
        Validate that a feedback type is one of the allowed types.
        Raises ValidationError if invalid.
        """
        allowed_types = {'concept_validation', 'primitive_validation', 'solution_guidance'}
        if feedback_type not in allowed_types:
            raise ValidationError(f"Invalid feedback type. Must be one of: {allowed_types}")
