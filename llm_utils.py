# llm_utils.py
import os
from ollama import Client
import signal
import re # Import the regular expression module

# Dictionary to cache Ollama client instances
_llm_clients = {}

def get_llm_client(model_name: str, host: str = 'http://localhost:11434'):
    """
    Retrieves or creates an Ollama client instance for a given model.
    Caches clients to avoid re-initialization.

    Args:
        model_name (str): The name of the Ollama model (e.g., 'llama2', 'mistral').
        host (str): The Ollama host URL. Defaults to 'http://localhost:11434'.

    Returns:
        ollama.Client: An initialized Ollama client.
    """
    if model_name not in _llm_clients:
        try:
            # Initialize the client with the specified host
            client = Client(host=host)
            # Verify the model exists by attempting to list models (optional, but good for early error detection)
            # This might be slow if there are many models, so consider if strictly necessary for minimal version.
            # models = client.list()
            # if not any(m['name'].startswith(model_name) for m in models['models']):
            #     print(f"Warning: Model '{model_name}' not found on Ollama server at {host}. Please ensure it's pulled.")
            client.model = model_name # Store the model name on the client for easier access
            _llm_clients[model_name] = client
            print(f"Initialized Ollama client for model: {model_name} at {host}")
        except Exception as e:
            print(f"Error initializing Ollama client for {model_name} at {host}: {e}")
            raise
    return _llm_clients[model_name]

def call_llm(llm_client: Client, prompt: str, system_message: str = None,
             temperature: float = 0.7, num_predict: int = 1024, **kwargs) -> str:
    """
    Makes a chat completion call to the Ollama LLM and cleans the response.

    Args:
        llm_client (ollama.Client): The initialized Ollama client.
        prompt (str): The user's prompt.
        system_message (str, optional): A system message to guide the LLM. Defaults to None.
        temperature (float, optional): Controls the randomness of the output. Defaults to 0.7.
        num_predict (int, optional): 
        **kwargs: Additional arguments to pass to the client.chat method (e.g., 'options').

    Returns:
        str: The generated text from the LLM, with <think> tags removed and empty lines removed,
             or an empty string if an error occurs.
    """
    messages = []
    if system_message:
        messages.append({'role': 'system', 'content': system_message})
    messages.append({'role': 'user', 'content': prompt})

    # Ollama's chat method expects temperature within 'options'
    options = kwargs.pop('options', {}) # Get existing options or create a new dict
    options['temperature'] = temperature # Add temperature to options
    options["num_predict"] = num_predict
   
    def handler(signum, frame):
      raise TimeoutError("Ollama generation timed out!")

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(30)  # 30-second timeout

    try:
        # Ensure the model is passed, as it's stored on the client object
        response = llm_client.chat(model=llm_client.model, messages=messages, options=options, **kwargs)
        signal.alarm(0)
        raw_content = response['message']['content']

        # Remove the <think>...</think> section using a regular expression
        if "<think>" in raw_content and "</think>" in raw_content:
            cleaned_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL)
        else:
            cleaned_content = raw_content
        
        # Remove empty lines and strip whitespace from each line
        lines = cleaned_content.splitlines()
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        # print("--- PROMPT ---")
        # print(prompt)
        # print("--- RAW RESPONSe ---")
        # print(raw_content)
        # print("--- CLEANED ---")
        # print("\n".join(non_empty_lines))

        # Join the non-empty lines back with a single newline character
        return "\n".join(non_empty_lines)

    except TimeoutError as e:
      print('Timeout occurred - model may have hung.')

    except Exception as e:
        print(f"LLM call failed for model {llm_client.model}: {e}")
        return "" # Return empty string on failure to prevent crashes
