from os import environ
environ['USER_AGENT'] = "RAG_APPLICATION"
environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
environ['CUDA_VISIBLE_DEVICES'] = "0"


import gradio as gr
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import platform
import rag.preprocessing as preprocessing
import re
import torch


def process_documents(message):
    docs_list = list()
    provided_docs = dict()
    # If URL(s), process each url and add them to the doc list
    if message["text"]:
        url_list = message["text"].split(",")
        docs = [WebBaseLoader(url).load() for url in url_list]
        docs_list += [item for sublist in docs for item in sublist]
        provided_docs["websites"] = url_list

    # If file(s), process each file and add it to the doc list
    if message["files"]:
        provided_files = list()
        for file in message["files"]:
            if platform.system() == "Linux":
                filename = file.split("/")[-1]
            elif platform.system() == "Windows":
                filename = file.split("\\")[-1]
            # JSON
            if re.fullmatch(r".*\.(json)$", filename):              
                print("JSON support is in the works")
                continue                
            # PDF
            elif re.fullmatch(r".*\.(pdf)$", filename):         
                docs = [PyPDFLoader(file).load()]
                docs_list += [item for sublist in docs for item in sublist]
                provided_files.append(filename)
            # Text (.md and .txt)
            elif re.fullmatch(r".*\.(md|txt)$", filename):              
                docs = [TextLoader(file).load()]
                docs_list += [item for sublist in docs for item in sublist]
                provided_files.append(filename)
            else:
                print(f"{filename}'s file format is not supported yet.")

        provided_docs["files"] = [provided_files]

    # Chunk and embed docs_list
    doc_splits = preprocessing.chunk_documents(docs_list)
    global retriever
    try:
        retriever = preprocessing.embed_doc_splits(doc_splits)
    except torch.OutOfMemoryError:
        print("\nGPU out of memory.\n\nRestart gradio_app.py")

    # Print the documents that were provided        
    print(f"Documents provided: {provided_docs}")


# Define prompt templates
prompt_templates = {
    "Default Template": """You are an assistant for question-answering tasks.
    - Use only the following documents to answer the question.
    - If you don't know the answer, simply say: "I don't know."
    - Use UP TO five sentences maximum and keep the answer concise.
Question: {question}
Documents: {documents}
Answer:""",

    "Concise Template": """You are an assistant designed to extract factual information from documents.
    - Given the following context: {documents}, provide a concise, accurate answer to the question: {question}
Answer:""",

    "Detailed Template": """You are an assistant designed to provide clear and detailed answers.
    - Use the documents provided to form your response.
    - Aim for clarity and explain the answer in detail, breaking down concepts where necessary.
    - If the information is unclear or you are missing information, say that.
    - Provide a detailed response, but do not hallucinate.
Question: {question}
Documents: {documents}
Answer:""",

    "Step-by-step Template": """You are an assistant designed to break down complex tasks or processes.
    - Based on the provided documents, explain the steps required to {question}. 
    - Be sure to include all relevant details from the context provided.
Documents: {documents}
Answer:""",

    "Summarization Template": """ You are an assistant designed to summarize key points from documents.
    - Separate the main ideas and conclusions from the following documents into sections.
    - Summarize each section under it's own header.
    - Format your summarization with markdown file (.md) formatting.
Documents: {documents}
Answer:""",
}

def update_template(selected_template):
    global prompt_template
    prompt_template = prompt_templates[selected_template]
    # Return the content of the selected template
    return prompt_template


def generate_response(message, history):
    try:
        if not retriever:
            pass
    except NameError: 
        response = gr.ChatMessage(role="assistant", content="Provide documents above before I can answer that")
        return response
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "documents"],
    )
    
    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
    )

    rag_chain = prompt | llm | StrOutputParser()

    try:
        # Retrieve relevant documents
        documents = retriever.invoke(message)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
    except ValueError as e:
        response = gr.ChatMessage(
            role="assistant",
            content="Input documents before you can ask questions."
            )
        return response
    
    # Get the answer from the LLM
    response = rag_chain.invoke({"question": message, "documents": doc_texts})
    
    return response


with gr.Blocks() as app:
    with gr.Column():
        # Docs
        doc_input = gr.MultimodalTextbox(
            interactive=True,
            file_count="multiple",
            placeholder="Upload files and/or enter URLs separated by a comma.",
            label="Document Input",
            show_label=True,
        )
        doc_submit = doc_input.submit(
            process_documents,
            [doc_input],
            [doc_input]
        )

        # Prompt Templates
        default_template = list(prompt_templates.keys())[0]  # Set default to first template
        template_dropdown = gr.Dropdown(
            choices=list(prompt_templates.keys()),
            label="Select a Prompt Template",
            interactive=True,
            value=default_template  # Set default value
        )
        template_display = gr.Textbox(
            label="Selected Template:",
            interactive=False,
            lines=5
        )

        # Update the displayed template when a new one is selected
        template_dropdown.change(
            fn=update_template,
            inputs=[template_dropdown],
            outputs=[template_display]
        )

    
    # Chatbot
    chatbot = gr.Chatbot(height=650, type="messages")  # Increase the height of the chatbot
    chat_interface = gr.ChatInterface(
        generate_response,
        chatbot=chatbot,
        type="messages",
        show_progress="full",
        )

    # Set initial value for template_display
    app.load(
        fn=update_template,
        inputs=[template_dropdown],
        outputs=[template_display]
    )


if __name__ == "__main__":
    app.launch()
