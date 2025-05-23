# fitness_functions/all_ones.py
from typing import Any
from ollama import Client # For type hinting
from llm_utils import get_llm_client, call_llm # Import utility functions

class FitnessEvaluator:
    """
    A fitness evaluator for the "print 1s" task.
    It evaluates how many '1's an LLM generates in response to a given prompt.
    """
    def __init__(self, task_description: str, llm_model_name: str, ollama_host: str, temperature: float):
        """
        Initializes the FitnessEvaluator.

        Args:
            task_description (str): The high-level task description (e.g., "print 1s").
                                    Used here for context, though not directly in get_fitness for this simple task.
            llm_model_name (str): The Ollama model to use for evaluating prompts.
            ollama_host (str): The host URL for the Ollama server.
            temperature (float): The temperature for LLM calls during evaluation.
        """
        self.task_description = task_description
        self.llm_model_name = llm_model_name
        self.ollama_host = ollama_host
        self.temperature = temperature
        self.llm_client = None # Will be initialized lazily or in new_generation

        print(f"FitnessEvaluator initialized for task: '{self.task_description}'")
        print(f"  Using LLM model: {self.llm_model_name} for evaluation.")

    def new_generation(self):
        """
        Called at the beginning of each new generation.
        For this simple task, it ensures the LLM client is ready.
        For more complex tasks, this could involve loading new test data batches.
        """
        if self.llm_client is None:
            try:
                self.llm_client = get_llm_client(self.llm_model_name, self.ollama_host)
                print(f"FitnessEvaluator LLM client for '{self.llm_model_name}' is ready.")
            except Exception as e:
                print(f"Error initializing fitness evaluator's LLM client: {e}")
                self.llm_client = None # Ensure it's None on failure
                raise # Re-raise to signal a critical error

        # In a more complex scenario, you might load a new batch of evaluation data here
        # self.current_evaluation_data = load_random_batch_for_task(self.task_description)
        pass # No specific setup needed for 'all_ones' beyond LLM client

    def get_fitness(self, prompt: str) -> int:
        """
        Evaluates the fitness of a given prompt for the "print 1s" task.
        The fitness is the count of '1's in the LLM's response to the prompt.

        Args:
            prompt (str): The prompt to evaluate.

        Returns:
            int: The fitness score (number of '1's). Returns 0 on error or if LLM client is not ready.
        """
        if self.llm_client is None:
            print("Error: FitnessEvaluator LLM client not initialized. Cannot evaluate prompt.")
            return 0

        # The prompt itself contains the instruction for the LLM
        # For "print 1s", the prompt might be "Print 10 ones"
        # The LLM's response is then evaluated.
        try:
            llm_response = call_llm(
                llm_client=self.llm_client,
                prompt=prompt + " Keep response to 100 characters.",
                system_message="Your task is to follow the instruction precisely.",
                temperature=self.temperature # Use the dedicated fitness temperature
            )
            fitness_score = llm_response.count("1")
            print("--- EVAL PROMPT ---")
            print(prompt)
            print(f"--- RESPONSE (fit={fitness_score}) ---")
            print(llm_response)
            # Count the occurrences of '1' in the LLM's response
            return fitness_score
        except Exception as e:
            print(f"Error during fitness evaluation for prompt '{prompt[:50]}...': {e}")
            return 0 # Return 0 fitness on error

# This block is for direct testing of the FitnessEvaluator class
if __name__ == '__main__':
    print("--- Testing fitness_functions/all_ones.py ---")
    
    # Ensure Ollama server is running and qwen3:0.6b model is pulled
    test_task = "print 1s"
    test_model = "qwen3:0.6b" # Use your specified model
    test_host = "http://localhost:11434"
    test_temp = 0.1

    try:
        evaluator = FitnessEvaluator(
            task_description=test_task,
            llm_model_name=test_model,
            ollama_host=test_host,
            temperature=test_temp
        )
        evaluator.new_generation() # Simulate start of a new generation

        test_prompt_good = "Generate a string containing exactly 15 ones. Only output the ones."
        test_prompt_bad = "Generate a string containing only zeros."
        test_prompt_mixed = "Print 5 ones and 3 zeros."

        print(f"\nEvaluating prompt: '{test_prompt_good}'")
        score_good = evaluator.get_fitness(test_prompt_good)
        print(f"  Fitness: {score_good}")

        print(f"\nEvaluating prompt: '{test_prompt_bad}'")
        score_bad = evaluator.get_fitness(test_prompt_bad)
        print(f"  Fitness: {score_bad}")

        print(f"\nEvaluating prompt: '{test_prompt_mixed}'")
        score_mixed = evaluator.get_fitness(test_prompt_mixed)
        print(f"  Fitness: {score_mixed}")

    except Exception as e:
        print(f"Could not run FitnessEvaluator test. Ensure Ollama server is running and '{test_model}' is pulled. Error: {e}")
