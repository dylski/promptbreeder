# thinking_styles.py
import random
import re
from typing import List
from ollama import Client # For type hinting, though actual client comes from llm_utils

# Import the LLM utility functions
from llm_utils import call_llm

# A set of default thinking styles, which instruct the LLM on how to approach a problem
# or how to interpret a prompt. These are inspired by various prompting techniques.
DEFAULT_THINKING_STYLES = [
    "Think step-by-step and explain your reasoning at each stage.",
    "Adopt the persona of a highly critical editor, looking for flaws and ambiguities.",
    "Act as a creative brainstorming partner, generating diverse and unconventional ideas.",
    "Simulate a logical deduction process, ensuring all conclusions follow strictly from premises.",
    "Imagine you are a seasoned expert in the relevant domain, providing authoritative insights.",
    "Consider the ethical implications of the request and your response.",
    "Focus on simplifying complex concepts into easily understandable terms.",
    "Prioritize efficiency and conciseness in your output.",
    "Emphasize thoroughness and provide comprehensive details.",
    "Approach the problem from multiple perspectives before settling on a solution.",
    "Break down the problem into smaller, manageable sub-problems.",
    "Evaluate the pros and cons of different approaches before giving an answer.",
    "Think like a devil's advocate, challenging assumptions and common beliefs.",
    "Adopt a pedagogical approach, aiming to teach the user the underlying principles.",
    "Consider the user's potential intent and tailor the response accordingly.",
    "Generate a list of questions that would help clarify the instruction.",
    "Think about potential edge cases or unusual scenarios.",
    "Focus on providing actionable advice or practical solutions.",
    "Prioritize user experience and ease of interaction in your response.",
    "Synthesize information from various sources to form a coherent answer."
]

def get_random_thinking_style() -> str:
    """
    Returns a random thinking style prompt from the default list.
    """
    return random.choice(DEFAULT_THINKING_STYLES)

def generate_new_thinking_styles(
    llm_client: Client,
    num_styles: int = 10,
    base_instruction: str = "create a list of diverse prompts that will instruct an LLM on a specific thinking style or approach to a problem. Output the result as a python list without any preamble or markdown.",
    system_message: str = "You are a prompt engineering assistant. Your goal is to generate clear and actionable instructions for LLMs regarding their thinking process. Always output only the Python list and nothing else.",
    temperature: float = 0.8
) -> List[str]:
    """
    Generates a list of new thinking styles using an LLM.

    Args:
        llm_client (ollama.Client): The initialized Ollama client to use for generation.
        num_styles (int): The number of new thinking styles to generate.
        base_instruction (str): The core instruction for the LLM to generate thinking styles.
        system_message (str): The system message to guide the LLM's role.
        temperature (float): The temperature for LLM generation, controlling creativity.

    Returns:
        List[str]: A list of newly generated thinking styles.
    """
    print(f"Generating {num_styles} new thinking styles using LLM model: {llm_client.model}...")
    # Emphasize that only the list should be outputted
    prompt_generation_request = f"{base_instruction}\nGenerate exactly {num_styles} distinct thinking styles. Ensure each style is a string element in the list. Do not include any other text, just the Python list. Start the output directly with '['."

    # Call the LLM to generate the prompts
    raw_response = call_llm(
        llm_client=llm_client,
        prompt=prompt_generation_request,
        system_message=system_message,
        temperature=temperature
    )

    generated_styles = []
    try:
        # Use regex to find the Python list pattern in the raw response
        match = re.search(r'\[\s*(?:\"[^"]*\"(?:\s*,\s*\"[^"]*\")*)*\s*\]', raw_response, re.DOTALL)
        if not match:
            match = re.search(r"\[\s*(?:\'[^']*\'(?:\s*,\s*\'[^']*\')*)*\s*\]", raw_response, re.DOTALL)
        
        if match:
            list_string = match.group(0)
            # Safely evaluate the extracted string as a Python list
            parsed_list = eval(list_string)
            if isinstance(parsed_list, list):
                generated_styles = [str(s) for s in parsed_list if isinstance(s, str)]
                print(f"Successfully generated {len(generated_styles)} new thinking styles.")
            else:
                print(f"Warning: Extracted content was not a list after parsing. Extracted: {list_string[:200]}...")
        else:
            print(f"Warning: Could not find a Python list pattern in the LLM response. Raw response: {raw_response[:200]}...")
    except Exception as e:
        print(f"Error parsing generated thinking styles: {e}. Raw response: {raw_response[:200]}...")

    return generated_styles

if __name__ == '__main__':
    # This block is for testing purposes if you run this file directly
    print("--- Testing thinking_styles.py ---")
    print("\nDefault Thinking Styles:")
    for i, style in enumerate(DEFAULT_THINKING_STYLES):
        print(f"{i+1}. {style}")

    print(f"\nRandom Thinking Style: {get_random_thinking_style()}")

    # Example of generating new styles (requires an Ollama server running and a model pulled)
    try:
        from llm_utils import get_llm_client # Import here for __main__ block
        test_llm_client = get_llm_client(model_name='qwen3:0.6b', host='http://localhost:11434')

        print("\nAttempting to generate 5 new thinking styles...")
        new_styles = generate_new_thinking_styles(test_llm_client, num_styles=5)
        if new_styles:
            print("\nGenerated New Thinking Styles:")
            for i, style in enumerate(new_styles):
                print(f"{i+1}. {style}")
        else:
            print("No new thinking styles were generated.")
    except Exception as e:
        print(f"Could not run LLM generation test. Ensure Ollama server is running and model is pulled. Error: {e}")

