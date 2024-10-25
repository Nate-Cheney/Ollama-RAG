# About
This is a personal project which allows me to provide documents to an LLM running with Ollama.

# Setup
- Install Ollama and a Llama3.1 model
- Install the following Python libraries:

  ```
  pip install langchain langchain_community langchain_huggingface scikit-learn langchain-ollama
  ```
- Clone the Ollama-RAG repository

# Use ollama-rag
### Start
- Start the Llama3.1 model
  ```
  Ollama run Llama3.1
  ```
- Run `ollama-rag.py` from the terminal
  ```
  Python ./ollama-rag.py
  ```

### Usage
Once the ollama-rag.py script has been ran:
1. Document Sources 
-   There are 3 possible document source types.
-   It is possible to select 1 or more options.
    ```
    1, 2, 3  # One source of each type
    ```
-   It is possible to repeat options.
    ```
    1,1,2  # Two files and one directory
    ```

2. Prompt Selection
-   The predefined prompts allow for an easy way to provide context and instructions prior to questioning.

3. Questions
-   The LLM will use the prompt selected in step 2 as instructions for how to handle the question.
-   After a question has been answered the user will have a chance to ask more.
