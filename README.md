# Minimal Promptbreeder Implementation

This repository contains a minimal Python implementation of the Promptbreeder concept, a self-referential self-improvement method for Large Language Models (LLMs) that evolves prompts for a given domain. Inspired by the [Promptbreeder paper](https://arxiv.org/pdf/2309.16797), this project demonstrates the core ideas of using a genetic algorithm to evolve task-specific prompts and their associated mutation instructions.

## Features

* **Genetic Algorithm Core:** Implements a binary tournament genetic algorithm to evolve a population of prompts.
* **Coupled Prompt Evolution:** Each "individual" in the population consists of a `(task_prompt, mutation_prompt)` pair, allowing the mutation mechanism itself to evolve alongside the task prompts.
* **Hypermutation:** Includes a mechanism where the `mutation_prompt` itself can be mutated (hypermutation). This mutation is conditionally accepted based on a "quick check" of its effectiveness.
* **Zeroth-Order Mutation:** Allows for generating entirely new task prompts from scratch based on the overall task description.
* **First-Order Mutation:** Modifies an existing task prompt using its coupled `mutation_prompt`.
* **Dynamic Fitness Evaluation:** Supports loading external Python files as `FitnessEvaluator` classes, enabling flexible and task-specific fitness scoring.
* **Ollama Integration:** Uses `ollama-python` for local LLM inference, making it easy to run experiments with various models.
* **Results Logging:** Saves detailed generation-wise data (population, fitness, best prompts) to a JSON file for analysis.

---

## Prerequisites

Before running this project, ensure you have:

1.  **Python 3.8+:** Installed on your system.
2.  **Ollama:** Installed and running. Download from [ollama.com](https://ollama.com/).
3.  **Ollama Models:** Pull the necessary LLM models. The default models used in this project are `qwen3:0.6b`. You can pull them using:
    ```bash
    ollama pull qwen3:0.6b
    ```

---

## Setup

1.  **Clone the repository (or create the files):**
    ```bash
    # If you're creating files manually, ensure they are in the correct structure
    # promptbreeder_project/
    # ├── ga.py
    # ├── llm_utils.py
    # ├── mutator.py
    # ├── mutation_prompts.py
    # ├── thinking_styles.py
    # ├── fitness_functions/
    # │   └── all_ones.py
    # └── README.md
    ```
2.  **Install Python dependencies:**
    ```bash
    pip install ollama
    ```

---

## Usage

To run the Promptbreeder GA, execute the `ga.py` script from your terminal.

```bash
python ga.py [OPTIONS]
```

**Command-Line Arguments:**

* --task <description> (Required): A high-level description of the task for which prompts are being evolved (e.g., "print 1s", "summarize news articles"). This helps guide initial prompt generation and zeroth-order mutations.
* --seed_prompt <prompt> (Default: "Print a sequence of ones."): An initial task prompt used as a fallback if LLM generation fails during population initialization.
* --max_gens <number> (Default: 5): The maximum number of generations the GA will run.
* --pop_size <number> (Default: 10): The size of the prompt population. It will be adjusted to an even number if an odd value is provided.
* --fitness_fn <path> (Required): The path to the Python file containing your FitnessEvaluator class (e.g., "fitness_functions/all_ones.py").
* --llm_ga_model <model_name> (Default: "qwen3:0.6b"): The Ollama model to use for GA operations (initial population generation, prompt mutation, hypermutation).
* --llm_fitness_model <model_name> (Default: "qwen3:0.6b"): The Ollama model to use for evaluating prompts within the fitness function.
* --ollama_host <url> (Default: "http://localhost:11434"): The host URL for your Ollama server.
* --temp_ga <float> (Default: 0.7): Temperature for LLM calls during GA operations (higher for more creativity in mutations).
* --temp_fitness <float> (Default: 0.1): Temperature for LLM calls within the fitness evaluator (lower for more deterministic and consistent evaluation).
* --output_file <path> (Default: "ga_results.json"): Path to the JSON file where all GA results (population, fitness per generation) will be saved.

```bash
python ga.py \
    --task="print 1s" \
    --seed_prompt="Generate a string of exactly 5 ones." \
    --max_gens=3 \
    --pop_size=4 \
    --fitness_fn="fitness_functions/all_ones.py" \
    --llm_ga_model="qwen3:0.6b" \
    --llm_fitness_model="qwen3:0.6b" \
    --output_file="my_ones_run_results.json"
```

## File Structure

* ga.py: The main script that orchestrates the Genetic Algorithm, including population management, evaluation, selection, and mutation.
* llm_utils.py: A utility module for interacting with the Ollama LLM client, handling client initialization, API calls, and response cleaning (removing <think> tags and empty lines).
* mutator.py: Contains the mutate_prompt function, which implements the core mutation logic for task prompts (zeroth-order and first-order) and hypermutation for mutation prompts.
* mutation_prompts.py: Defines the INITIAL_MUTATION_PROMPTS list, used to seed the coupled mutation prompts in the GA.
* thinking_styles.py: Contains a list of DEFAULT_THINKING_STYLES that guide the LLM's approach during mutation operations.
* fitness_functions/: A directory for custom fitness evaluation modules.
* fitness_functions/all_ones.py: An example FitnessEvaluator class that scores prompts based on how many '1's the LLM generates in response.

## Customizing Fitness Functions

To adapt the Promptbreeder to your specific task:

1. Create a new Python file in the fitness_functions/ directory (e.g., my_task_fitness.py).
2. Inside this file, define a class named FitnessEvaluator.
3. This class must have:
    * An __init__(self, task_description: str, llm_model_name: str, ollama_host: str, temperature: float) method to initialize your evaluator and its LLM client.
    * A get_fitness(self, prompt: str) -> int method that takes a task_prompt string and returns an integer fitness score.
    * An optional new_generation(self) method, which will be called at the start of each GA generation (useful for loading new test data batches, etc.).
4. Pass the path to your new fitness file using the --fitness_fn argument when running ga.py.

Example my_task_fitness.py structure:

```python
# fitness_functions/my_task_fitness.py
from typing import Any
from llm_utils import get_llm_client, call_llm

class FitnessEvaluator:
    def __init__(self, task_description: str, llm_model_name: str, ollama_host: str, temperature: float):
        self.task_description = task_description
        self.llm_client = get_llm_client(llm_model_name, ollama_host)
        self.temperature = temperature
        # Load your task-specific test data here
        # self.test_data = load_my_data(task_description)

    def new_generation(self):
        # Optional: Prepare for a new generation (e.g., sample new test batch)
        pass

    def get_fitness(self, prompt: str) -> int:
        # Use self.llm_client to evaluate the prompt against your task
        # For example, if it's a summarization task:
        # summary = call_llm(self.llm_client, f"{prompt}\nSummarize the following text: {self.test_data['article']}", temperature=self.temperature)
        # score = evaluate_summary_quality(summary, self.test_data['ground_truth'])
        # return score
        pass # Implement your actual fitness logic here
```
