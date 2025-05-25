# mutation_prompts.py
import random
from typing import List

# A set of initial mutation prompts, inspired by the Promptbreeder paper.
# These prompts instruct an LLM on how to modify an existing instruction (task prompt).
# In this flatter structure, these serve as the initial pool for the coupled mutation prompts.
INITIAL_MUTATION_PROMPTS = [
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

# The functions get_random_mutation_prompt and generate_new_mutation_prompts
# are being removed from this file, as their logic will be integrated into
# the mutator.py and ga.py for the new coupled evolution approach.

if __name__ == '__main__':
    # This block is for testing purposes if you run this file directly
    print("--- Testing mutation_prompts.py ---")
    print("\nInitial Mutation Prompts:")
    for i, prompt in enumerate(INITIAL_MUTATION_PROMPTS):
        print(f"{i+1}. {prompt}")
    print(f"\nTotal initial mutation prompts: {len(INITIAL_MUTATION_PROMPTS)}")

