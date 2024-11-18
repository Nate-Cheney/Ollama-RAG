from os import environ
environ['USER_AGENT'] = "RAG_APPLICATION"
environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
environ['CUDA_VISIBLE_DEVICES'] = "0"


from utils.audit import *
import gradio as gr
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import platform
import utils.prompt_templates as pt
import utils.preprocessing as preprocessing
import re
import torch
import utils.youtube as yt


def process_documents(message):
    docs_list = list()
    provided_docs = dict()
    # If URL(s), process each url and add them to the doc list
    if message["text"]:
        url_list = message["text"].split(",")
        provided_urls = list()
        provided_videos = list()
        transcript_list = list()
        for url in url_list:
            url = url.strip()
            if re.match("^(https://www.youtube.com/playlist)", url):
                playlist = yt.get_playlist_videos(url)
                for video in playlist:
                    video_title = yt.transcript_main(url=video)
                    transcript_list.append(video_title)
                    provided_videos.append(video_title)
                continue
            elif re.match("^(https://www.youtube.com/watch)", url) or re.match("^https://youtu.be/", url):
                video_title = yt.transcript_main(url=url)
                transcript_list.append(video_title)
                provided_videos.append(video_title)
                continue
            else:
                docs = [WebBaseLoader(url).load()]
                docs_list += [item for sublist in docs for item in sublist]
                provided_urls.append(url)

        if provided_urls:
            provided_docs["websites"] = provided_urls

        if provided_videos:
            provided_docs["youtube videos"] = provided_videos
            dir_path = r"transcripts"
            for file in os.listdir(dir_path):
                docs = [TextLoader(os.path.join(dir_path, file)).load()]
                docs_list += [item for sublist in docs for item in sublist]
                os.remove(os.path.join(dir_path, file))

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
                continue

            # Text (.md and .txt)
            elif re.fullmatch(r".*\.(md|txt)$", filename):              
                docs = [TextLoader(file).load()]
                docs_list += [item for sublist in docs for item in sublist]
                provided_files.append(filename)
                continue
            
            else:
                print(f"{filename}'s file format is not supported yet.")

        provided_docs["files"] = [provided_files]

    # Chunk and embed docs_list
    doc_splits = preprocessing.chunk_documents(docs_list)
    global retriever
    try:
        retriever = preprocessing.embed_doc_splits(doc_splits)

        grade_retrieved_documents(question=message)

    except torch.OutOfMemoryError:
        print("\nGPU out of memory.\n\nRestart gradio_app.py")

    # Print the documents that were provided        
    print(f"Documents provided: {provided_docs}")


def update_template(selected_template):
    global prompt_template
    prompt_template = pt.prompt_template_dict[selected_template]

    return prompt_template


def generate_response(message, history):
    try:
        if not retriever:
            pass
    except NameError: 
        response = gr.ChatMessage(role="assistant", content="Provide documents above before I can answer that")
        return response
    
    while True:
        try:
            prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>""" +\
                prompt_template +\
                """
    - List the source(s) of any documents used at the end of your response.
    - If the source is a file, ignore the path to the file include only the filename. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Documents: \n\n {documents} \n\n
    Question: {question}
    Answer: <|eot_id|><start_header_id|>assistant<|end_header_id|>"""
            
            print("\n" + prompt_template)
            prompt = PromptTemplate(
                template=prompt_template,
                input_variables=["question", "documents"],
            )
            break
        except NameError:
            prompt_template = pt.prompt_template_dict["Default Template"]

        llm = ChatOllama(
            model="llama3.1",
            temperature=0.1,
        )

    rag_chain = prompt | llm | StrOutputParser()

    # Retrieve relevant documents
    documents = retriever.invoke(message)
    # Format the retrieved documents with source metadata
    doc_texts = "\n".join(
        [f"{doc.page_content}\n(Source: {doc.metadata.get('source', 'Unknown')})" for doc in documents]
    )   
    
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
        default_template = list(pt.prompt_template_dict.keys())[0]  # Set default to first template
        template_dropdown = gr.Dropdown(
            choices=list(pt.prompt_template_dict.keys()),
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
    chatbot = gr.Chatbot(height=700, type="messages")  # Increase the height of the chatbot
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
