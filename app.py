from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

class OllamaRAG:
    def __init__(self, docs_list=list()):
        '''Get Documents'''

        '''Chunk Documents'''
        # Initialize a text splitter with specified chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=250, chunk_overlap=0
        )
        # Split the documents into chunks
        doc_splits = text_splitter.split_documents(docs_list)
        
        '''Embed Documents'''
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Create embeddings for documents and store them in a vector store
        self.vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=embeddings,
        )
        self.retriever = self.vectorstore.as_retriever(k=15)
        

    def grade_retrieved_documents(self, question):
        '''LLM'''
        llm = ChatOllama(model="llama3.1", format="json", temperature=0)

        prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance
            of a retrieved document to a user question. If the document contains keywords related to the user question,
            grade it as relevant. It does not neede to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score "yes" or "no" score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key "score" and no preamble or explaination.    
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Here is the retrieved document(s): \n\n {documents} \n\n
            Here is the user question: {question} \n <|eot_id|><start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["question", "documents"]
        )

        retrieval_grader = prompt | llm | JsonOutputParser()

        '''Retrieve Documents'''
        self.documents = self.retriever.invoke(question)

        '''Audit Documents'''
        grade = retrieval_grader.invoke({"question": question, "documents": self.documents})
        
        print(grade)
        return grade


    def generate_response(self, question):
        '''Generate response'''
        llm = ChatOllama(model="llama3.1", temperature=0)

        prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks.
            - Use only the following documents to answer the question.
            - If you don't know the answer, simply say: "I don't know."\n
            - Keep your answers as clear and consise as possible.\n
            - List the source(s) of any documents used at the end of your response.
            - If the source is a file, ignore the path to the file include only the filename.\n
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Documents: \n\n {documents} \n\n
            Question: \n\n {question} \n\n
            Answer: <|eot_id|><start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["question", "documents"]
        )

        rag_chain = prompt | llm | StrOutputParser()
    
        answer = rag_chain.invoke({"question": question, "documents": self.documents})

        print(answer)
        return answer


if __name__ == "__main__":
    # Init
    document = TextLoader(r"test_docs/1. Procedural Introduction to Cybersecurity Litigation.txt").load()
    q = "When was the mona lisa painted?"
    rag_agent = OllamaRAG(docs_list=document)

    # Grade document retrieval
    grade = rag_agent.grade_retrieved_documents(question=q)
    
    # Generate response
    if grade == {'score': 'yes'}:
        rag_agent.generate_response(question=q)
