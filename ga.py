# ga.py
import random
import importlib.util
import sys
from typing import List, Callable, Tuple, Any
from ollama import Client # For type hinting

# Import utility functions for LLM calls, mutation prompts, and thinking styles
from llm_utils import call_llm, get_llm_client
from mutation_prompts import get_random_mutation_prompt, generate_new_mutation_prompts
from thinking_styles import get_random_thinking_style, generate_new_thinking_styles

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
    temperature_fitness: float = 0.1 # Lower temp for more deterministic fitness evaluation
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
    """
    print(f"\n--- Starting Promptbreeder GA for task: '{task_description}' ---")
    print(f"Population Size: {pop_size}, Max Generations: {max_gens}")
    print(f"GA LLM Model: {llm_ga_model_name}, Fitness LLM Model: {llm_fitness_model_name}")

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
    population: List[str] = []
    #population.append(initial_seed_prompt) # Add the user-provided seed prompt

    print("\nGenerating initial population...")
    initial_gen_system_message = "You are a helpful assistant. Generate a concise and effective instruction."
    for i in range(pop_size):
        # Generate diverse initial prompts related to the task
        prompt_request = f"Generate a distinct and effective instruction related to the task: '{task_description}'. The instruction should be ready for an LLM to follow."
        generated_prompt = call_llm(
            llm_ga_client,
            prompt_request,
            system_message=initial_gen_system_message,
            temperature=temperature_ga
        )
        if generated_prompt:
            population.append(generated_prompt.strip())
        else:
            # Fallback if LLM fails to generate: duplicate seed or a default
            population.append(initial_seed_prompt + f" (variant {i+1})")
            print(f"Warning: LLM failed to generate initial prompt {i+1}, using fallback.")
    
    # Ensure population size is correct, even with potential LLM failures
    population = population[:pop_size]
    print(f"Initial population size: {len(population)}")
    for i, p in enumerate(population):
        print(f"{i+1}. {p}")

    # --- Genetic Algorithm Loop ---
    best_overall_prompt = initial_seed_prompt
    highest_overall_fitness = -float('inf')

    for gen in range(max_gens):
        print(f"\n--- Generation {gen + 1}/{max_gens} ---")
        
        # Call new_generation hook on the fitness evaluator
        try:
            fitness_evaluator.new_generation()
            print("Fitness evaluator prepared for new generation.")
        except AttributeError:
            # new_generation is optional, so handle if it doesn't exist
            pass
        except Exception as e:
            print(f"Error calling new_generation on fitness evaluator: {e}")

        evaluated_population: List[Tuple[str, int]] = [] # List of (prompt, fitness) tuples

        # 1. Evaluation
        print("Evaluating population fitness...")
        for i, prompt in enumerate(population):
            try:
                # Call the get_fitness method on the instantiated evaluator
                fitness_score = fitness_evaluator.get_fitness(prompt)
                evaluated_population.append((prompt, fitness_score))
                print(f"  Prompt {i+1} (Fitness: {fitness_score}): {prompt[:80]}...")
            except Exception as e:
                print(f"Error evaluating prompt '{prompt[:50]}...': {e}. Assigning fitness 0.")
                evaluated_population.append((prompt, 0)) # Assign 0 fitness on error

        # Sort by fitness (descending)
        evaluated_population.sort(key=lambda x: x[1], reverse=True)

        current_best_prompt, current_highest_fitness = evaluated_population[0]
        print(f"\nGeneration {gen + 1} Best Prompt (Fitness: {current_highest_fitness}):\n{current_best_prompt}")

        if current_highest_fitness > highest_overall_fitness:
            highest_overall_fitness = current_highest_fitness
            best_overall_prompt = current_best_prompt
            print(f"New overall best prompt found!")

        if gen == max_gens - 1:
            break # Last generation, no need to create next population

        # 2. Selection (Binary Tournament) and Mutation to create next generation
        new_population: List[str] = []
        
        # Elitism: Carry over the top 1 or 2 prompts directly
        num_elite = min(2, pop_size // 5, len(evaluated_population)) # Max 2 elite, or 20% of pop, or available
        for i in range(num_elite):
            new_population.append(evaluated_population[i][0])
        if num_elite > 0:
            print(f"Carrying over {num_elite} elite prompts.")

        print("Creating next generation through binary tournament and mutation...")
        while len(new_population) < pop_size:
            # Select two random individuals for tournament
            candidate1 = random.choice(evaluated_population)
            candidate2 = random.choice(evaluated_population)

            # Winner is the one with higher fitness
            winner_prompt = candidate1[0] if candidate1[1] >= candidate2[1] else candidate2[0]

            # Mutate the winner
            mutation_prompt = get_random_mutation_prompt()
            thinking_style = get_random_thinking_style()

            mutation_instruction = (
                f"{thinking_style}\n"
                f"{mutation_prompt}\n"
                f"Modify the following instruction based on the task '{task_description}':\n"
                f"Original instruction: \"{winner_prompt}\"\n"
                f"Output only the modified instruction, without any preamble or markdown."
            )
            
            mutated_prompt = call_llm(
                llm_ga_client,
                mutation_instruction,
                system_message="You are a creative and precise prompt modifier. Your task is to transform instructions as requested.",
                temperature=temperature_ga
            )

            if mutated_prompt:
                new_population.append(mutated_prompt.strip())
            else:
                # Fallback if mutation fails: add the winner again or duplicate an existing prompt
                new_population.append(winner_prompt)
                print(f"Warning: LLM failed to mutate prompt, adding winner again as fallback.")

        population = new_population[:pop_size] # Ensure exact population size

    print("\n--- GA Complete ---")
    for i, p in enumerate(population):
        print(f"{i+1}. {p}")
    print(f"\nOverall Best Prompt (Fitness: {highest_overall_fitness}):\n{best_overall_prompt}")
    return best_overall_prompt, highest_overall_fitness

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Run a minimal Promptbreeder Genetic Algorithm.")
    parser.add_argument("--task", type=str, required=True,
                        help="The high-level task description for which prompts are being evolved (e.g., 'print 1s').")
    parser.add_argument("--seed_prompt", type=str, default="Print a sequence of ones.",
                        help="The initial seed prompt to start the evolution.")
    parser.add_argument("--max_gens", type=int, default=5,
                        help="Maximum number of generations for the GA.")
    parser.add_argument("--pop_size", type=int, default=5,
                        help="Size of the prompt population.")
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

    args = parser.parse_args()

    # Example usage:
    # python ga.py --task="print 1s" --seed_prompt="Generate a string of ones." --max_gens=3 --pop_size=4 --fitness_fn="fitness_functions/all_ones.py" --llm_ga_model="qwen3:0.6b" --llm_fitness_model="qwen3:0.6b"

    best_prompt, best_fitness = run_ga(
        task_description=args.task,
        initial_seed_prompt=args.seed_prompt,
        llm_ga_model_name=args.llm_ga_model,
        llm_fitness_model_name=args.llm_fitness_model,
        fitness_function_path=args.fitness_fn,
        pop_size=args.pop_size,
        max_gens=args.max_gens,
        ollama_host=args.ollama_host,
        temperature_ga=args.temp_ga,
        temperature_fitness=args.temp_fitness
    )
