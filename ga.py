# ga.py
import random
import importlib.util
import sys
import json # Import json for saving data
from typing import List, Callable, Tuple, Any
from ollama import Client # For type hinting

# Import utility functions for LLM calls
from llm_utils import call_llm, get_llm_client
# Import the new mutate_prompt function
from mutator import mutate_prompt
# Import INITIAL_MUTATION_PROMPTS for initial population seeding
from mutation_prompts import INITIAL_MUTATION_PROMPTS


def load_fitness_evaluator_class(file_path: str) -> type:
    """
    Dynamically loads the 'FitnessEvaluator' class from a specified Python file.

    Args:
        file_path (str): The path to the Python file containing the FitnessEvaluator class.

    Returns:
        type: The loaded FitnessEvaluator class.

    Raises:
        FileNotFoundError: If the file does not exist.
        AttributeError: If the 'FitnessEvaluator' class is not found in the module.
    """
    spec = importlib.util.spec_from_file_location("fitness_module", file_path)
    if spec is None:
        raise FileNotFoundError(f"Could not find module specification for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["fitness_module"] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        print(f"Error executing module {file_path}: {e}")
        raise

    if not hasattr(module, 'FitnessEvaluator') or not isinstance(getattr(module, 'FitnessEvaluator'), type):
        raise AttributeError(f"The file {file_path} must define a class named 'FitnessEvaluator'.")

    print(f"Successfully loaded FitnessEvaluator class from: {file_path}")
    return module.FitnessEvaluator

def run_ga(
    task_description: str,
    initial_seed_prompt: str,
    llm_ga_model_name: str,
    llm_fitness_model_name: str,
    fitness_function_path: str,
    pop_size: int = 10,
    max_gens: int = 20,
    ollama_host: str = 'http://localhost:11434',
    temperature_ga: float = 0.7,
    temperature_fitness: float = 0.1, # Lower temp for more deterministic fitness evaluation
    output_file_path: str = "ga_results.json", # New parameter for output file
    max_prompt_length: int = 256 # New parameter for maximum prompt length
):
    """
    Runs the Genetic Algorithm for prompt evolution.

    Args:
        task_description (str): The high-level task for which prompts are being evolved.
                                This is passed to the fitness function's initialization.
        initial_seed_prompt (str): The starting prompt for the first generation.
        llm_ga_model_name (str): The name of the Ollama model to use for GA operations
                                 (initial population generation, mutation).
        llm_fitness_model_name (str): The name of the Ollama model to use for evaluating
                                      prompts in the fitness function.
        fitness_function_path (str): Path to the Python file containing the 'FitnessEvaluator' class.
        pop_size (int): The number of prompts in the population.
        max_gens (int): The maximum number of generations to run the GA.
        ollama_host (str): The host URL for the Ollama server.
        temperature_ga (float): Temperature for LLM calls during GA operations (mutation, initial gen).
        temperature_fitness (float): Temperature for LLM calls within the fitness evaluator.
        output_file_path (str): Path to the JSON file where GA results will be saved.
        max_prompt_length (int): Maximum length for generated prompts (passed to num_predict).
    """
    # Ensure population size is even for pairing
    if pop_size % 2 != 0:
        print(f"Warning: Population size {pop_size} is odd. Adjusting to {pop_size + 1} for pairing.")
        pop_size += 1

    print(f"\n--- Starting Promptbreeder GA for task: '{task_description}' ---")
    print(f"Population Size: {pop_size}, Max Generations: {max_gens}")
    print(f"GA LLM Model: {llm_ga_model_name}, Fitness LLM Model: {llm_fitness_model_name}")
    print(f"Results will be saved to: {output_file_path}")
    print(f"Max Prompt Length: {max_prompt_length} characters.")

    # Initialize LLM client for GA operations
    try:
        llm_ga_client = get_llm_client(llm_ga_model_name, ollama_host)
    except Exception as e:
        print(f"Failed to initialize GA LLM client: {e}")
        return

    # Load the FitnessEvaluator class and instantiate it
    try:
        FitnessEvaluatorClass = load_fitness_evaluator_class(fitness_function_path)
        # Pass necessary parameters to the FitnessEvaluator constructor
        fitness_evaluator = FitnessEvaluatorClass(
            task_description=task_description,
            llm_model_name=llm_fitness_model_name,
            ollama_host=ollama_host,
            temperature=temperature_fitness
        )
    except (FileNotFoundError, AttributeError, Exception) as e:
        print(f"Error loading or initializing FitnessEvaluator: {e}")
        return

    # --- Initial Population Generation ---
    # Population now stores tuples of (task_prompt, mutation_prompt)
    population: List[Tuple[str, str]] = []

    print("\nGenerating initial population...")
    initial_gen_system_message = "You are a helpful assistant. Generate a concise and effective instruction."
    for i in range(pop_size):
        # Generate a task prompt
        prompt_request = f"Generate a distinct and effective instruction related to the task: '{task_description}'. The instruction should be ready for an LLM to follow."
        generated_task_prompt = call_llm(
            llm_ga_client,
            prompt_request,
            system_message=initial_gen_system_message,
            temperature=temperature_ga,
            num_predict=max_prompt_length # Pass max_prompt_length here
        )
        # Choose an initial mutation prompt for this task prompt
        initial_mutation_prompt = random.choice(INITIAL_MUTATION_PROMPTS)

        if generated_task_prompt:
            population.append((generated_task_prompt.strip(), initial_mutation_prompt))
        else:
            # Fallback if LLM fails to generate: use the initial seed prompt with a random initial mutator
            population.append((initial_seed_prompt + f" (initial_gen_fallback_{i+1})", initial_mutation_prompt))
            print(f"Warning: LLM failed to generate initial task prompt {i+1}, using fallback.")

    # Ensure population size is correct, even with potential LLM failures
    population = population[:pop_size]
    print(f"Initial population size: {len(population)}")
    for i, (tp, mp) in enumerate(population):
        print(f"{i+1}. Task: {tp[:80]}... | Mutator: {mp[:50]}...")

    # --- Genetic Algorithm Loop ---
    best_overall_task_prompt = initial_seed_prompt
    best_overall_mutation_prompt = random.choice(INITIAL_MUTATION_PROMPTS) # Initialize with a random one
    highest_overall_fitness = -float('inf') # Changed to float
    all_generations_data = [] # List to store data for all generations

    for gen in range(max_gens):
        print(f"\n--- Generation {gen + 1}/{max_gens} ---")

        # Call new_generation hook on the fitness evaluator, passing the current population
        try:
            fitness_evaluator.new_generation(population) # <<-- IMPORTANT CHANGE HERE: pass population
            print("Fitness evaluator prepared for new generation.")
        except AttributeError:
            # new_generation is optional, so handle if it doesn't exist
            pass
        except Exception as e:
            print(f"Error calling new_generation on fitness evaluator: {e}")

        # evaluated_population now stores (task_prompt, mutation_prompt, fitness)
        evaluated_population: List[Tuple[str, str, float]] = [] # Changed fitness type to float

        # 1. Evaluation
        print("Evaluating population fitness...")
        for i, (task_prompt, mutation_prompt) in enumerate(population):
            try:
                # Call the get_fitness method on the instantiated evaluator with the task prompt
                fitness_score = fitness_evaluator.get_fitness(task_prompt) # Will now return float
                evaluated_population.append((task_prompt, mutation_prompt, fitness_score))
                print(f"  Prompt {i+1} (Fitness: {fitness_score:.4f}): Task: {task_prompt[:80]}... | Mutator: {mutation_prompt[:50]}...") # Format for display
            except Exception as e:
                print(f"Error evaluating prompt '{task_prompt[:50]}...': {e}. Assigning fitness 0.0.") # Changed to 0.0
                evaluated_population.append((task_prompt, mutation_prompt, 0.0)) # Assign 0.0 fitness on error

        # Sort by fitness (descending)
        evaluated_population.sort(key=lambda x: x[2], reverse=True)

        current_best_task_prompt, current_best_mutation_prompt, current_highest_fitness = evaluated_population[0]
        print(f"\nGeneration {gen + 1} Best Prompt (Fitness: {current_highest_fitness:.4f}):") # Format for display
        print(f"  Task: {current_best_task_prompt}")
        print(f"  Mutator: {current_best_mutation_prompt}")


        if current_highest_fitness > highest_overall_fitness:
            highest_overall_fitness = current_highest_fitness
            best_overall_task_prompt = current_best_task_prompt
            best_overall_mutation_prompt = current_best_mutation_prompt
            print(f"New overall best prompt found!")

        # Store generation data
        generation_data = {
            "generation": gen + 1,
            "population_size": len(population),
            "current_population": [{"task_prompt": tp, "mutation_prompt": mp} for tp, mp in population],
            "evaluated_population": [{"task_prompt": tp, "mutation_prompt": mp, "fitness": f} for tp, mp, f in evaluated_population],
            "best_prompt_this_gen": {"task_prompt": current_best_task_prompt, "mutation_prompt": current_best_mutation_prompt},
            "highest_fitness_this_gen": current_highest_fitness
        }
        all_generations_data.append(generation_data)

        if gen == max_gens - 1:
            break # Last generation, no need to create next population

        # 2. Selection (Pairing and Winners) and Mutation to create next generation
        new_population: List[Tuple[str, str]] = []

        # Shuffle the evaluated population to create random pairs for tournament
        random.shuffle(evaluated_population)

        print("Creating next generation through pairing, binary tournament, parent copying, and mutation...")
        # Iterate through pairs
        for i in range(0, pop_size, 2):
            # Ensure we don't go out of bounds if pop_size was adjusted or odd (though we force even)
            if i + 1 >= len(evaluated_population):
                break

            candidate1 = evaluated_population[i] # (task_prompt, mutation_prompt, fitness)
            candidate2 = evaluated_population[i+1] # (task_prompt, mutation_prompt, fitness)

            # Winner is the one with higher fitness
            winner_entry = candidate1 if candidate1[2] >= candidate2[2] else candidate2 # winner_entry is (tp, mp, f)
            winner_task_prompt = winner_entry[0]
            winner_mutation_prompt = winner_entry[1]
            winner_fitness = winner_entry[2] # Get winner's fitness for quick check in mutator

            # 1. Copy the winner's (task_prompt, mutation_prompt) pair directly to the new population
            new_population.append((winner_task_prompt, winner_mutation_prompt))

            # 2. Mutate the winner to create one offspring using the mutator module
            mutated_task_prompt, mutated_mutation_prompt = mutate_prompt(
                llm_client=llm_ga_client,
                current_task_prompt=winner_task_prompt,
                current_mutation_prompt=winner_mutation_prompt,
                current_fitness=winner_fitness, # Pass winner's fitness for quick check
                task_description=task_description,
                fitness_evaluator=fitness_evaluator, # Pass the fitness evaluator for quick check
                temperature_ga=temperature_ga,
                temperature_fitness=temperature_fitness, # Pass fitness temp for quick check
                max_prompt_length=max_prompt_length # Pass max_prompt_length here
            )

            # The mutate_prompt function already handles returning original on failure and stripping
            new_population.append((mutated_task_prompt, mutated_mutation_prompt))
            if mutated_task_prompt == winner_task_prompt and mutated_mutation_prompt == winner_mutation_prompt:
                print(f"  Note: Mutation of '{winner_task_prompt[:50]}...' failed, used winner as offspring fallback.")


        population = new_population[:pop_size] # Ensure exact population size if there were fallbacks

    print("\n--- GA Complete ---")
    print("\nFinal Population:")
    for i, (tp, mp) in enumerate(population):
        print(f"{i+1}. Task: {tp[:80]}... | Mutator: {mp[:50]}...")
    print(f"\nOverall Best Prompt (Fitness: {highest_overall_fitness:.4f}):") # Format for display
    print(f"  Task Prompt: {best_overall_task_prompt}")
    print(f"  Mutation Prompt: {best_overall_mutation_prompt}")


    # Save all generations data to a JSON file
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(all_generations_data, f, indent=4, ensure_ascii=False)
        print(f"GA results saved to {output_file_path}")
    except Exception as e:
        print(f"Error saving GA results to {output_file_path}: {e}")

    return best_overall_task_prompt, best_overall_mutation_prompt, highest_overall_fitness

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run a minimal Promptbreeder Genetic Algorithm.")
    parser.add_argument("--task", type=str, required=True,
                        help="The high-level task description for which prompts are being evolved (e.g., 'print 1s').")
    parser.add_argument("--seed_prompt", type=str, default="Print a sequence of ones.",
                        help="The initial seed prompt used as a fallback if LLM generation fails.")
    parser.add_argument("--max_gens", type=int, default=5,
                        help="Maximum number of generations for the GA.")
    parser.add_argument("--pop_size", type=int, default=10,
                        help="Size of the prompt population (will be adjusted to even if odd).")
    parser.add_argument("--fitness_fn", type=str, required=True,
                        help="Path to the Python file containing the 'FitnessEvaluator' class (e.g., 'fitness_functions/all_ones.py').")
    parser.add_argument("--llm_ga_model", type=str, default="qwen3:0.6b",
                        help="Ollama model name for GA operations (initial population, mutation).")
    parser.add_argument("--llm_fitness_model", type=str, default="qwen3:0.6b",
                        help="Ollama model name for fitness evaluation.")
    parser.add_argument("--ollama_host", type=str, default="http://localhost:11434",
                        help="Ollama server host URL.")
    parser.add_argument("--temp_ga", type=float, default=0.7,
                        help="Temperature for LLM calls during GA operations (mutation, initial gen).")
    parser.add_argument("--temp_fitness", type=float, default=0.1,
                        help="Temperature for LLM calls within the fitness evaluator (lower for determinism).")
    parser.add_argument("--output_file", type=str, default="ga_results.json",
                        help="Path to the JSON file to save GA results.")
    parser.add_argument("--max_prompt_length", type=int, default=256,
                        help="Maximum length for generated prompts (number of tokens/characters).")

    args = parser.parse_args()

    # Example usage with the new diversity fitness function:
    # python ga.py --task="diverse short stories" --seed_prompt="Write a simple story." --max_gens=3 --pop_size=4 --fitness_fn="fitness_functions/diversity_fitness.py" --llm_ga_model="qwen3:0.6b" --llm_fitness_model="qwen3:0.6b" --output_file="my_diversity_run.json"

    best_task_prompt, best_mutation_prompt, best_fitness = run_ga(
        task_description=args.task,
        initial_seed_prompt=args.seed_prompt,
        llm_ga_model_name=args.llm_ga_model,
        llm_fitness_model_name=args.llm_fitness_model,
        fitness_function_path=args.fitness_fn,
        pop_size=args.pop_size,
        max_gens=args.max_gens,
        ollama_host=args.ollama_host,
        temperature_ga=args.temp_ga,
        temperature_fitness=args.temp_fitness,
        output_file_path=args.output_file,
        max_prompt_length=args.max_prompt_length
    )

