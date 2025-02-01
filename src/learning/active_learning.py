import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.llm.llm_interface import LLMInterface
from src.storage.learning_state import LearningStateManager

logger = logging.getLogger(__name__)

class ActiveLearningSystem:
    """System for active learning through hypothesis generation and testing"""
    
    def __init__(self, llm: LLMInterface, state_manager: LearningStateManager):
        self.llm = llm
        self.state_manager = state_manager
        self.hypotheses = []
        self.experiments = []
        
    async def form_hypothesis(self, observations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Form a hypothesis based on observations"""
        # Prepare observation data for LLM
        observation_text = "\n".join([
            f"Observation {i+1}:\n" + 
            f"Input: {obs.get('input')}\n" +
            f"Output: {obs.get('output')}\n" +
            f"Pattern: {obs.get('pattern')}\n"
            for i, obs in enumerate(observations)
        ])
        
        # Generate hypothesis using LLM
        prompt = f"""
        Based on these observations:
        {observation_text}
        
        Please form a hypothesis that:
        1. Explains the observed patterns
        2. Makes testable predictions
        3. Can be validated or refuted
        
        Provide your response in JSON format with:
        - A clear hypothesis statement
        - Your confidence level (0-1)
        - Supporting evidence
        - Suggested test cases
        """
        
        hypothesis = await self.llm.propose_experiment({"prompt": prompt})
        
        # Record hypothesis
        self.hypotheses.append({
            "timestamp": datetime.now().isoformat(),
            "hypothesis": hypothesis,
            "observations": observations
        })
        
        # Save to state manager
        self.state_manager.record_pattern({
            "type": "hypothesis",
            "data": hypothesis,
            "context": {"observations": len(observations)}
        })
        
        return hypothesis
        
    async def validate_hypothesis(self, hypothesis: Dict[str, Any], examples: List[Dict[str, Any]]) -> bool:
        """Validate a hypothesis against examples"""
        # Extract test cases from hypothesis
        test_cases = hypothesis.get("test_cases", [])
        
        # Run validation
        total_tests = len(test_cases)
        passed_tests = 0
        
        for test_case in test_cases:
            # Check if any example matches the test case
            for example in examples:
                if self._matches_test_case(example, test_case):
                    passed_tests += 1
                    break
                    
        # Calculate validation score
        validation_score = passed_tests / total_tests if total_tests > 0 else 0
        
        # Record validation results
        validation_result = {
            "timestamp": datetime.now().isoformat(),
            "hypothesis": hypothesis,
            "validation_score": validation_score,
            "tests_passed": passed_tests,
            "total_tests": total_tests
        }
        
        # Save to state manager
        self.state_manager.record_pattern({
            "type": "validation",
            "data": validation_result,
            "success": validation_score > 0.8
        })
        
        # Request human feedback if results are inconclusive
        if 0.4 <= validation_score <= 0.8:
            await self.llm.request_feedback(
                hypothesis=hypothesis["hypothesis"],
                confidence=validation_score,
                context=f"Validation results: {passed_tests}/{total_tests} tests passed"
            )
            
        return validation_score > 0.8
        
    async def generate_test_cases(self, concept: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate test cases to validate a concept"""
        # Use LLM to generate test cases
        experiment = await self.llm.propose_experiment(concept)
        
        # Record experiment
        self.experiments.append({
            "timestamp": datetime.now().isoformat(),
            "concept": concept,
            "experiment": experiment
        })
        
        # Save to state manager
        self.state_manager.record_pattern({
            "type": "experiment",
            "data": experiment,
            "context": {"concept": concept.get("name")}
        })
        
        return experiment.get("test_cases", [])
        
    def _matches_test_case(self, example: Dict[str, Any], test_case: Dict[str, Any]) -> bool:
        """Check if an example matches a test case"""
        try:
            input_match = example.get("input") == test_case.get("input")
            output_match = example.get("output") == test_case.get("expected_output")
            return input_match and output_match
        except Exception as e:
            logger.error(f"Error comparing example to test case: {e}")
            return False
