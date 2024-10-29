# About
This is a personal project which allows me to provide documents to an LLM running with Ollama.

# Setup
- Install Ollama and a Llama3.1 model
- Install the following Python libraries:

  ```
  pip install gradio langchain langchain_community langchain_huggingface langchain-ollama scikit-learn
  ```
- Clone the Ollama-RAG repository

# Use ollama-rag
## Gradio
### Start
- Start the Llama3.1 model
  ```
  Ollama run Llama3.1
  ```
- Run `gradio_app.py` from the terminal
  ```
  Python ./gradio_app.py
  ```
- Go to [http://127.0.0.1:7860](http://127.0.0.1:7860) with a web browser

### Usage
- Input documents or paste URLs separated by commas into the top input box.

- Allow the provided docs to be processed.

- Ask away.

## Terminal
### Start
- Start the Llama3.1 model
  ```
  Ollama run Llama3.1
  ```
- Run `terminal_app.py` from the terminal
  ```
  Python ./terminal_app.py
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
