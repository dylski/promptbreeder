# mutator.py
from ollama import Client # For type hinting
from llm_utils import call_llm
from mutation_prompts import get_random_mutation_prompt
from thinking_styles import get_random_thinking_style

def mutate_prompt(
    llm_client: Client,
    original_prompt: str,
    task_description: str,
    temperature: float
) -> str:
    """
    Mutates a given prompt using an LLM.

    Args:
        llm_client (ollama.Client): The initialized Ollama client for GA operations.
        original_prompt (str): The prompt to be mutated.
        task_description (str): The high-level task description, used to guide the mutation.
        temperature (float): The temperature for the LLM call during mutation.

    Returns:
        str: The mutated prompt, or the original prompt if mutation fails.
    """
    mutation_prompt_instruction = get_random_mutation_prompt()
    thinking_style_instruction = get_random_thinking_style()

    mutation_instruction = (
        f"{thinking_style_instruction}\n"
        f"{mutation_prompt_instruction}\n"
        f"Modify the following instruction based on the task '{task_description}':\n"
        f"Original instruction: \"{original_prompt}\"\n"
        f"Output only the modified instruction, without any preamble or markdown."
    )

    mutated_prompt = call_llm(
        llm_client,
        mutation_instruction,
        system_message="You are a creative and precise prompt modifier. Your task is to transform instructions as requested.",
        temperature=temperature
    )

    if mutated_prompt:
        return mutated_prompt.strip()
    else:
        print(f"Warning: LLM failed to mutate prompt '{original_prompt[:50]}...', returning original prompt as fallback.")
        return original_prompt # Return original prompt as fallback if mutation fails

# This block is for direct testing of the mutator function
if __name__ == '__main__':
    print("--- Testing mutator.py ---")

    test_model = "qwen3:0.6b" # Use your specified model
    test_host = "http://localhost:11434"
    test_temp = 0.7

    try:
        # Initialize a dummy client for testing purposes
        test_llm_client = Client(host=test_host)
        test_llm_client.model = test_model # Manually set the model name for the client

        test_original_prompt = "Generate a list of 5 random numbers."
        test_task_description = "generate lists of numbers"

        print(f"\nOriginal Prompt: '{test_original_prompt}'")
        print(f"Task: '{test_task_description}'")

        mutated = mutate_prompt(test_llm_client, test_original_prompt, test_task_description, test_temp)
        print(f"\nMutated Prompt (attempt 1): '{mutated}'")

        mutated = mutate_prompt(test_llm_client, mutated, test_task_description, test_temp)
        print(f"Mutated Prompt (attempt 2): '{mutated}'")

    except Exception as e:
        print(f"Could not run mutator test. Ensure Ollama server is running and '{test_model}' is pulled. Error: {e}")

