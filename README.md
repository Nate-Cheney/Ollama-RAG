# About
This is a personal project which allows me to provide documents to an LLM running with Ollama.

# Setup
- Install Ollama and a Llama3.1 model
- Install the following Python libraries:

  ```
  pip install -r requirements.txt
  ```
- Clone the Ollama-RAG repository

# Use ollama-rag (gradio)
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
- Input documents or paste URLs separated by commas into the input box.

- Allow for the provided docs to be processed.

- Ask away.

