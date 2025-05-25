# mutator.py
import random
from typing import List, Tuple, Any
from ollama import Client # For type hinting
from llm_utils import call_llm
from mutation_prompts import INITIAL_MUTATION_PROMPTS # To seed the coupled mutation prompts
from thinking_styles import get_random_thinking_style

# Probability of performing hypermutation (mutating the mutation prompt itself)
HYPERMUTATION_PROB = 0.15 # 15% chance to mutate the mutation prompt
# Probability of creating a new mutation prompt from scratch during hypermutation
# (as opposed to mutating the existing one)
NEW_MUTATOR_PROMPT_PROB = 0.2 # 20% chance to generate a new mutator prompt from scratch during hypermutation

# Probability of performing zeroth-order mutation on the task prompt
ZEROTH_ORDER_PROB = 0.5 # 50% chance for zeroth-order vs. first-order task prompt mutation

def mutate_prompt(
    llm_client: Client,
    current_task_prompt: str,
    current_mutation_prompt: str,
    current_fitness: float, # Added this parameter to be used in hypermutation quick check
    task_description: str,
    fitness_evaluator: Any, # Instance of FitnessEvaluator from ga.py
    temperature_ga: float,
    temperature_fitness: float,
    max_prompt_length: int # New parameter for maximum prompt length
) -> Tuple[str, str]:
    """
    Mutates either the task prompt or its coupled mutation prompt.
    If task prompt mutation, it can be zeroth-order or first-order.

    Args:
        llm_client (ollama.Client): The initialized Ollama client for GA operations.
        current_task_prompt (str): The task prompt to be potentially mutated.
        current_mutation_prompt (str): The mutation prompt coupled with the task prompt.
        current_fitness (float): The fitness of the current_task_prompt. Used for hypermutation quick check.
        task_description (str): The high-level task description, used to guide mutations.
        fitness_evaluator (Any): An instance of the main GA's FitnessEvaluator for quick checks.
        temperature_ga (float): Temperature for LLM calls during GA operations (mutation of task/mutator prompts).
        temperature_fitness (float): Temperature for LLM calls within the quick check for hypermutation.
        max_prompt_length (int): Maximum length for generated prompts (passed to num_predict).

    Returns:
        Tuple[str, str]: A tuple containing the new (task_prompt, mutation_prompt).
                         Returns original pair if mutation fails.
    """
    new_task_prompt = current_task_prompt
    new_mutation_prompt = current_mutation_prompt

    # Decide whether to mutate the task prompt or the mutation prompt (hypermutation)
    if random.random() < HYPERMUTATION_PROB:
        # --- Perform Hypermutation (mutate the mutation prompt) ---
        print(f"  Performing Hypermutation on mutation prompt: '{current_mutation_prompt[:50]}...'")

        # original_mp_fitness is now directly from current_fitness of the task prompt
        # because the quick check evaluates how well the MP generates a *task prompt*
        # that performs. The current task prompt's fitness is our baseline.
        original_mp_fitness_baseline = current_fitness
        mutated_mp_candidate = ""

        # Option 1: Create a new mutator prompt from scratch (like initial population)
        if random.random() < NEW_MUTATOR_PROMPT_PROB:
            print("    Generating new mutation prompt from scratch.")
            mp_gen_instruction = (
                "Generate a distinct and effective instruction that will instruct an LLM to take an existing instruction "
                "and change it into a related instruction. Output only the instruction, without any preamble or markdown."
            )
            mutated_mp_candidate = call_llm(
                llm_client,
                mp_gen_instruction,
                system_message="You are a prompt engineering assistant. Generate a concise and actionable prompt mutation instruction.",
                temperature=temperature_ga,
                num_predict=max_prompt_length # Pass max_prompt_length here
            ).strip()
        else:
            # Option 2: Mutate the existing mutator prompt
            print("    Mutating existing mutation prompt.")
            thinking_style = get_random_thinking_style()
            hypermutation_instruction = (
                f"{thinking_style}\n"
                f"Please summarize and improve the following instruction for mutating prompts. "
                f"Make it more effective or diverse for transforming other prompts.\n"
                f"Original instruction: \"{current_mutation_prompt}\"\n"
                f"Output only the modified instruction, without any preamble or markdown."
            )
            mutated_mp_candidate = call_llm(
                llm_client,
                hypermutation_instruction,
                system_message="You are a meta-prompt modifier. Your task is to creatively alter instructions that guide other prompts.",
                temperature=temperature_ga,
                num_predict=max_prompt_length # Pass max_prompt_length here
            ).strip()

        if not mutated_mp_candidate:
            print("    Hypermutation failed to generate a candidate. Falling back to standard task prompt mutation.")
            # Fall through to the standard task prompt mutation if no candidate is generated
            pass # This will cause the code to continue to the 'else' block below

        else: # Only proceed with quick check if a candidate was generated
            # --- Quick Check for Hypermutation ---
            # Evaluate the effectiveness of the newly generated/mutated mutation prompt candidate.
            # We do this by seeing how well it mutates the *current_task_prompt* and checking its fitness.
            print(f"    Original MP test fitness baseline: {original_mp_fitness_baseline}")
            temp_task_prompt_mutated_mp = "" # Initialize outside try block
            mutated_mp_fitness = 0
            try:
                # Removed 'based on the task' as current_task_prompt should provide context
                temp_task_prompt_mutated_mp = call_llm(
                    llm_client=llm_client,
                    prompt=(
                        f"{mutated_mp_candidate}\n"
                        f"Modify the following instruction:\n"
                        f"Original instruction: \"{current_task_prompt}\"\n"
                        f"Output only the modified instruction, without any preamble or markdown."
                    ),
                    system_message="You are a creative and precise prompt modifier. Your task is to transform instructions as requested.",
                    temperature=temperature_ga, # Use GA temp for this internal mutation
                    num_predict=max_prompt_length # Pass max_prompt_length here
                )
                temp_task_prompt_mutated_mp = temp_task_prompt_mutated_mp.strip()
                mutated_mp_fitness = fitness_evaluator.get_fitness(temp_task_prompt_mutated_mp)
                print(f"    Mutated MP test fitness: {mutated_mp_fitness}")
            except Exception as e:
                print(f"    Error during mutated MP quick check: {e}. Assuming fitness 0.")
                mutated_mp_fitness = 0

            # If the mutated mutation prompt performs same or better, keep it AND its generated task prompt
            if mutated_mp_fitness >= original_mp_fitness_baseline:
                new_mutation_prompt = mutated_mp_candidate
                new_task_prompt = temp_task_prompt_mutated_mp # Use the task prompt generated during quick check
                print(f"    Hypermutation successful! New MP: '{new_mutation_prompt[:50]}...'")
                return new_task_prompt, new_mutation_prompt
            else:
                print("    Hypermutation led to worse performance. Falling back to standard task prompt mutation.")
                # new_mutation_prompt remains current_mutation_prompt, and we fall through
                # to the standard task prompt mutation below.
                pass

    # --- Standard Task Prompt Mutation (50:50 chance for Zeroth-order vs. First-order) ---
    # This block is executed if hypermutation was not chosen, or if it failed/led to worse performance.
    if random.random() < ZEROTH_ORDER_PROB:
        # --- Zeroth-order Mutation ---
        print("  Performing Zeroth-order task prompt mutation.")
        # Generate a new task prompt from scratch based on the overall task description.
        # The task_description is crucial here to ensure the generated prompt is relevant
        # to the overall goal of the GA, as there's no 'current_task_prompt' to derive context from.
        zeroth_order_instruction = (
            f"Generate a distinct and effective instruction related to the task: '{task_description}'. "
            f"The instruction should be ready for an LLM to follow. "
            f"Output only the instruction, without any preamble or markdown."
        )
        mutated_task_prompt_result = call_llm(
            llm_client,
            zeroth_order_instruction,
            system_message="You are a helpful assistant. Generate a concise and effective instruction.",
            temperature=temperature_ga,
            num_predict=max_prompt_length # Pass max_prompt_length here
        )
        if mutated_task_prompt_result:
            new_task_prompt = mutated_task_prompt_result.strip()
        else:
            print(f"  Warning: Zeroth-order task prompt mutation failed. Keeping original task prompt.")
            new_task_prompt = current_task_prompt # Fallback
    else:
        # --- First-order Mutation (existing logic) ---
        print("  Performing First-order task prompt mutation.")
        mutation_instruction = (
            f"{new_mutation_prompt}\n" # Use the (potentially original) mutation prompt
            f"Modify the following instruction:\n" # Removed 'based on the task'
            f"Original instruction: \"{current_task_prompt}\"\n"
            f"Output only the modified instruction, without any preamble or markdown."
        )

        mutated_task_prompt_result = call_llm(
            llm_client,
            mutation_instruction,
            system_message="You are a creative and precise prompt modifier. Your task is to transform instructions as requested.",
            temperature=temperature_ga,
            num_predict=max_prompt_length # Pass max_prompt_length here
        )

        if mutated_task_prompt_result:
            new_task_prompt = mutated_task_prompt_result.strip()
        else:
            print(f"  Warning: First-order task prompt mutation failed. Keeping original task prompt.")
            new_task_prompt = current_task_prompt # Fallback

    return new_task_prompt, new_mutation_prompt

# This block is for direct testing of the mutator functionality
if __name__ == '__main__':
    print("--- Testing mutator.py ---")

    test_llm_model = "qwen3:0.6b"
    test_ollama_host = "http://localhost:11434"
    test_temp_ga = 0.7
    test_temp_fitness = 0.1
    test_task = "Generate a short, positive sentence."

    # Dummy FitnessEvaluator for testing purposes (similar to the one in all_ones.py)
    class DummyFitnessEvaluator:
        def __init__(self, task_description, llm_model_name, ollama_host, temperature):
            print(f"DummyFitnessEvaluator initialized for task: {task_description}")
            self.llm_client = get_llm_client(llm_model_name, ollama_host)
            self.temperature = temperature
            self.task_description = task_description

        def get_fitness(self, prompt: str) -> int:
            # Simulate fitness: longer positive sentences get higher score
            try:
                response = call_llm(
                    llm_client=self.llm_client,
                    prompt=prompt + " Keep response to 100 characters.",
                    system_message="Your task is to follow the instruction precisely.",
                    temperature=self.temperature
                )
                fitness = response.count("1") # Reusing the all_ones logic for consistency
                # print(f"    Dummy Eval: '{response[:50]}...' Fitness: {fitness}")
                return fitness
            except Exception as e:
                print(f"    Dummy Eval Error: {e}. Returning 0 fitness.")
                return 0

        def new_generation(self):
            pass # No specific setup needed for dummy

    try:
        main_ga_llm_client = get_llm_client(test_llm_model, test_ollama_host)
        dummy_fitness_evaluator = DummyFitnessEvaluator(test_task, test_llm_model, test_ollama_host, test_temp_fitness)
        dummy_fitness_evaluator.new_generation() # Ensure LLM client is ready

        test_original_task_prompt = "Generate a sequence of ten '1's."
        test_original_mutation_prompt = random.choice(INITIAL_MUTATION_PROMPTS) # Pick a random initial one
        # Get an initial fitness for the quick check baseline
        initial_fitness_for_test = dummy_fitness_evaluator.get_fitness(test_original_task_prompt)


        print(f"\nOriginal Task Prompt: '{test_original_task_prompt}'")
        print(f"Original Mutation Prompt: '{test_original_mutation_prompt}'")
        print(f"Original Fitness: {initial_fitness_for_test}")
        print(f"Task: '{test_task}'")

        print("\n--- Performing a mutation (might be hypermutation or zeroth/first-order) ---")
        mutated_tp, mutated_mp = mutate_prompt(
            llm_client=main_ga_llm_client,
            current_task_prompt=test_original_task_prompt,
            current_mutation_prompt=test_original_mutation_prompt,
            current_fitness=initial_fitness_for_test, # Pass the initial fitness
            task_description=test_task,
            fitness_evaluator=dummy_fitness_evaluator,
            temperature_ga=test_temp_ga,
            temperature_fitness=test_temp_fitness,
            max_prompt_length=256 # Pass a dummy max_prompt_length for testing
        )
        print(f"\nResulting Task Prompt: '{mutated_tp}'")
        print(f"Resulting Mutation Prompt: '{mutated_mp}'")

        if mutated_mp != test_original_mutation_prompt:
            print("\nNOTE: Hypermutation occurred and was successful (mutation prompt changed)!")
        elif mutated_tp != test_original_task_prompt:
            print("\nNOTE: Task prompt was mutated (either zeroth-order or first-order).")
        else:
            print("\nNOTE: No effective mutation occurred (fallback to original prompts).")


    except Exception as e:
        print(f"Could not run mutator test. Ensure Ollama server is running and '{test_llm_model}' is pulled. Error: {e}")

