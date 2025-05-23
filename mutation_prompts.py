# mutation_prompts.py
import random
import re
from typing import List
from ollama import Client # For type hinting, though actual client comes from llm_utils

# Import the LLM utility functions
from llm_utils import call_llm

# A set of default mutation prompts, inspired by the Promptbreeder paper.
# These prompts instruct an LLM on how to modify an existing instruction (task prompt).
DEFAULT_MUTATION_PROMPTS = [
    "Refine the instruction to be more concise and direct.",
    "Expand the instruction to include more detailed constraints and requirements.",
    "Rephrase the instruction to target a different audience or skill level.",
    "Add a new example to the instruction that clarifies its intent.",
    "Remove an unnecessary example or constraint from the instruction.",
    "Make the instruction more specific by adding a numerical value or a concrete noun.",
    "Generalize the instruction by removing specific details and making it broader.",
    "Change the tone of the instruction to be more formal and academic.",
    "Change the tone of the instruction to be more informal and conversational.",
    "Introduce a new negative constraint, specifying what the output should NOT include.",
    "Add a positive constraint, specifying what the output MUST include.",
    "Ask the LLM to consider an alternative perspective or approach in its response.",
    "Instruct the LLM to break down the task into smaller, sequential steps.",
    "Tell the LLM to justify its reasoning for the output it provides.",
    "Suggest a creative or unusual way to interpret the instruction.",
    "Simplify the language of the instruction for easier understanding.",
    "Make the instruction more challenging by adding a complex condition.",
    "Add a time-based constraint, e.g., 'respond within 5 minutes'.",
    "Specify an output format, e.g., 'output as a JSON object'.",
    "Combine this instruction with another related instruction to form a compound task."
]

def get_random_mutation_prompt() -> str:
    """
    Returns a random mutation prompt from the default list.
    """
    return random.choice(DEFAULT_MUTATION_PROMPTS)

def generate_new_mutation_prompts(
    llm_client: Client,
    num_prompts: int = 10,
    base_instruction: str = "create a list of diverse prompts that will instruct an LLM to take an instruction and change it into a related instruction. Output the result as a python list without any preamble or markdown.",
    system_message: str = "You are a prompt engineering assistant. Your goal is to generate clear and actionable instructions for modifying other prompts. Always output only the Python list and nothing else.",
    temperature: float = 0.8
) -> List[str]:
    """
    Generates a list of new mutation prompts using an LLM.

    Args:
        llm_client (ollama.Client): The initialized Ollama client to use for generation.
        num_prompts (int): The number of new prompts to generate.
        base_instruction (str): The core instruction for the LLM to generate mutation prompts.
        system_message (str): The system message to guide the LLM's role.
        temperature (float): The temperature for LLM generation, controlling creativity.

    Returns:
        List[str]: A list of newly generated mutation prompts.
    """
    print(f"Generating {num_prompts} new mutation prompts using LLM model: {llm_client.model}...")
    # Emphasize that only the list should be outputted
    prompt_generation_request = f"{base_instruction}\nGenerate exactly {num_prompts} distinct prompts. Ensure each prompt is a string element in the list. Do not include any other text, just the Python list. Start the output directly with '['."

    # Call the LLM to generate the prompts
    raw_response = call_llm(
        llm_client=llm_client,
        prompt=prompt_generation_request,
        system_message=system_message,
        temperature=temperature
    )

    generated_prompts = []
    try:
        # Use regex to find the Python list pattern in the raw response
        # This is more robust than a direct eval() if the LLM adds extra text.
        # It looks for content starting with '[' and ending with ']'
        match = re.search(r'\[\s*(?:\"[^"]*\"(?:\s*,\s*\"[^"]*\")*)*\s*\]', raw_response, re.DOTALL)
        if not match:
            match = re.search(r"\[\s*(?:\'[^']*\'(?:\s*,\s*\'[^']*\')*)*\s*\]", raw_response, re.DOTALL)
        
        if match:
            list_string = match.group(0)
            # Safely evaluate the extracted string as a Python list
            parsed_list = eval(list_string)
            if isinstance(parsed_list, list):
                generated_prompts = [str(p) for p in parsed_list if isinstance(p, str)]
                print(f"Successfully generated {len(generated_prompts)} new mutation prompts.")
            else:
                print(f"Warning: Extracted content was not a list after parsing. Extracted: {list_string[:200]}...")
        else:
            print(f"Warning: Could not find a Python list pattern in the LLM response. Raw response: {raw_response[:200]}...")
    except Exception as e:
        print(f"Error parsing generated mutation prompts: {e}. Raw response: {raw_response[:200]}...")

    return generated_prompts

if __name__ == '__main__':
    # This block is for testing purposes if you run this file directly
    print("--- Testing mutation_prompts.py ---")
    print("\nDefault Mutation Prompts:")
    for i, prompt in enumerate(DEFAULT_MUTATION_PROMPTS):
        print(f"{i+1}. {prompt}")

    print(f"\nRandom Mutation Prompt: {get_random_mutation_prompt()}")

    # Example of generating new prompts (requires an Ollama server running and a model pulled)
    try:
        # For testing, you might need to adjust the model_name and host
        # Make sure you have 'llama2' or another model pulled via `ollama pull llama2`
        from llm_utils import get_llm_client # Import here for __main__ block
        test_llm_client = get_llm_client(model_name='qwen3:0.6b', host='http://localhost:11434') # Use the utility function

        print("\nAttempting to generate 5 new mutation prompts...")
        new_prompts = generate_new_mutation_prompts(test_llm_client, num_prompts=5)
        if new_prompts:
            print("\nGenerated New Mutation Prompts:")
            for i, prompt in enumerate(new_prompts):
                print(f"{i+1}. {prompt}")
        else:
            print("No new mutation prompts were generated.")
    except Exception as e:
        print(f"Could not run LLM generation test. Ensure Ollama server is running and model is pulled. Error: {e}")

