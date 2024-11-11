def chunk_documents(docs_list: list):
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    # Initialize a text splitter with specified chunk size and overlap
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    # Split the documents into chunks
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits


def embed_doc_splits(doc_splits):
    from langchain_community.vectorstores import SKLearnVectorStore
    from langchain_huggingface import HuggingFaceEmbeddings

    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create embeddings for documents and store them in a vector store
    vectorstore = SKLearnVectorStore.from_documents(
        documents=doc_splits,
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever(k=4)

    return retriever
