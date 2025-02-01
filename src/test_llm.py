import asyncio
import os
from dotenv import load_dotenv
from src.llm.llm_interface import LLMInterface
from src.learning.learning_orchestrator import LearningOrchestrator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_test():
    # Load environment variables from .env
    load_dotenv()
    
    # Initialize LLM interface with API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    llm = LLMInterface(api_key)
    orchestrator = LearningOrchestrator(llm)
    
    # Test task with simple pattern
    test_task = {
        "train": [
            {
                "input": [
                    [0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]
                ],
                "output": [
                    [1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]
                ]
            },
            {
                "input": [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                    [0, 0, 0, 0]
                ],
                "output": [
                    [1, 1, 1, 1],
                    [1, 0, 0, 1],
                    [1, 0, 0, 1],
                    [1, 1, 1, 1]
                ]
            }
        ]
    }
    
    try:
        logger.info("Starting test with simple inversion pattern...")
        result = await orchestrator.learn_from_task(test_task)
        logger.info(f"Learning result: {result}")
        
        # Test error handling with invalid task
        logger.info("\nTesting error handling with invalid task...")
        invalid_task = {
            "train": [
                {
                    "input": [[0, 0], [0, 1]],
                    # Missing output
                }
            ]
        }
        try:
            result = await orchestrator.learn_from_task(invalid_task)
        except Exception as e:
            logger.info(f"Successfully caught error for invalid task: {str(e)}")
        
        # Test retry logic with malformed prompt
        logger.info("\nTesting retry logic with malformed prompt...")
        malformed_prompt = "This is an intentionally malformed prompt that should trigger retries"
        try:
            response = await llm.get_completion(malformed_prompt, schema=llm.TRANSFORMATION_SCHEMA)
            logger.info(f"Response from malformed prompt: {response}")
        except Exception as e:
            logger.info(f"Successfully caught error for malformed prompt: {str(e)}")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(run_test())
