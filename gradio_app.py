from os import environ
environ['USER_AGENT'] = "RAG_APPLICATION"
environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
environ['CUDA_VISIBLE_DEVICES'] = "0"

import gradio as gr
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import rag.documents as documents
import rag.preprocessing as preprocessing
import time
import torch
import os


def process_documents(message):
    docs_list = list()
    # If URL(s), process each url and add them to the doc list
    if message["text"]:
        url_list = message["text"].split(",")
        docs = [WebBaseLoader(url).load() for url in url_list]
        docs_list += [item for sublist in docs for item in sublist]

    # If file(s), process each file and add it to the doc list
    if message["files"]:
        for file in message["files"]:
            docs = [TextLoader(file).load()]
            docs_list += [item for sublist in docs for item in sublist]

    # Chunk and embed docs_list
    doc_splits = preprocessing.chunk_documents(docs_list)
    global retriever
    try:
        retriever = preprocessing.embed_doc_splits(doc_splits)
    except torch.OutOfMemoryError:
        print("\nGPU out of memory.\n\nRestart gradio_app.py")


def generate_response(message, history):
    try:
        if not retriever:
            pass
    except NameError: 
        response = gr.ChatMessage(role="assistant", content="Provide documents above before I can answer that")
        return response
    
    prompt = PromptTemplate(
        template="""
        You are an assistant for question-answering tasks.
        Use the following documents to answer the question.
        If you don't know the answer, just say that you don't know.
        Use UP TO four sentences maximum and keep the answer concise:
        Question: {question}
        Documents: {documents}
        Answer:
        """,
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
            content="I don't have enough information to answer that question. Provide more documents."
            )
        return response
    
    # Get the answer from the LLM
    response = rag_chain.invoke({"question": message, "documents": doc_texts})
    
    return response


with gr.Blocks() as app:
    with gr.Row():
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

    # Chatbot
    chatbot = gr.Chatbot(height=800, type="messages")  # Increase the height of the chatbot
    chat_interface = gr.ChatInterface(
        generate_response,
        chatbot=chatbot,
        type="messages",
        show_progress="full",
        )


if __name__ == "__main__":
    app.launch()
