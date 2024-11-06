# About
Ollama-RAG allows for retrieval-augmented generation using llama3.1 and user provided documents (currently in the form of URLs, .MDs and .TXTs).

Ollama-RAG is a personal project and I can't guaruntee that this will work perfectly, but feel free to reach out with any questions, comments, or concerns.

# Setup
- Install Ollama and a Llama3.1 model
- Install the required Python libraries:

  ```
  # Install packages on Linux
  pip install -r requirements.txt
  ```

  ```
  # Install packages on Windows
  pip install -r win_requirements.txt

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
  python3 ./gradio_app.py
  ```
- Go to [http://127.0.0.1:7860](http://127.0.0.1:7860) with a web browser

### Usage
- Input documents or paste URLs separated by commas into the input box.

- Select one of the prompt templates.

- Make sure that the provided docs have already been processed.

- Ask away.

# Customization
Ollama-RAG should work with all of Ollama modles. **Please know that I have only tested it with llama3.1**.

To use a different model, change the model name in `gradio_app.py` line 109 to the desired model.

```
108    llm = ChatOllama(
109        model="llama3.1",  # replace the model here
110        temperature=0,
111    )
```
