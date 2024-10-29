import gradio as gr


def process_message(history, message):
    '''
    Documents
    '''
    from langchain_community.document_loaders import TextLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    docs = list()
    for file in message["files"]:
        # Show the files in the chat history
        history.append({"role": "user", "content": {"path": file}})
        docs.append(TextLoader(file).load())

    # flattens the list of lists (docs) into a single list (docs_list)
    docs_list = [item for sublist in docs for item in sublist]  

    # Initialize a text splitter
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
    )
    # Split the documents into chunks
    doc_splits = text_splitter.split_documents(docs_list)

    '''
    Embeddings
    '''
    from langchain_community.vectorstores import SKLearnVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings

    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create embeddings for documents and store them in a vector store
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
    )
    global retriever
    retriever = vectorstore.as_retriever(k=4)
    
    if message["text"] is not None:
            # Show the message in the chat history
            history.append({"role": "user", "content": message["text"]})

    return history, gr.MultimodalTextbox(value=None, interactive=False)
        

def generate_response(history, message):
    from langchain_ollama import ChatOllama
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Define the prompt template for the LLM
    prompt = PromptTemplate(
        template="""You are an assistant for question-answering tasks.
        Use the following documents to fill in the blanks.
        If you don't know the answer, just say that you don't know.
        Keep the answer concise:
        Question: {question}
        Documents: {documents}
        Answer:
        """,
        input_variables=["question", "documents"],
    )

    # Initialize the LLM with Llama 3.1 model
    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
    )

    rag_chain = prompt | llm | StrOutputParser()

    # Retrieve relevant documents
    documents = retriever.invoke(message["text"])
    # Extract content from retrieved documents
    doc_texts = "\\n".join([doc.page_content for doc in documents])

    # Get the answer from the LLM
    response = rag_chain.invoke({"question": message["text"], "documents": doc_texts})
    
     # Write the response to the chat
    history.append({"role": "assistant", "content": response})
    
    # Return both history and an empty MultimodalTextbox
    return history, gr.MultimodalTextbox(value=None, interactive=True)


with gr.Blocks() as app:
    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")

    chat_input = gr.MultimodalTextbox(
        interactive=True,
        file_count="multiple",
        placeholder="Upload one or more files and ask a question...",
        show_label=False,
    )

    chat_msg = chat_input.submit(
        process_message, 
        [chatbot, chat_input], 
        [chatbot, chat_input]
    )
    
    bot_msg = chat_msg.then(
        generate_response,
        [chatbot, chat_input],
        [chatbot, chat_input],
        api_name="bot_response"
    )



app.launch()
