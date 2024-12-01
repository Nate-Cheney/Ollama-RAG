from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
import openparse
import os
import utils.youtube as yt


class DocumentProcessor:
    '''The DocumentProcessor class holds the logic for converting media to a markdown formatted document.'''
    def __init__(self):
        self.docs_list = list()

    def pdf_processor(self, path: str):
        '''Note that PDF parsing is not finished. using the unitable parsing algorithm is the best option from OpenParse. I'd like to compare the OpenParse parser to the Ollama one.'''
        parser = openparse.DocumentParser(
                table_args={
                    "parsing_algorithm": "unitable",
                    "min_table_confidence": 0.8,
                },
        )    
        parsed_basic_doc = parser.parse(path)

        for node in parsed_basic_doc.nodes:
            print(node.model_dump(warnings="none")["text"])

    def text_processor(self, path: str):
        pass

    def web_processor(self, url: str):
        pass

    def youtube_processor(self, url: str):
        video_title = yt.transcript_main(url=url)

        llm = ChatOllama(model="llama3.1", temperature=0)

        prompt = PromptTemplate(
            template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant that transforms YouTube transcripts from Python dictionaries into markdown. 
            Use the context from the "text", "start", and "duration" to format the transcript.\n
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            Dictionary Transcript: \n\n {transcript} \n\n
            Markdown Transcript: <|eot_id|><start_header_id|>assistant<|end_header_id|>
            """,
            input_variables=["transcript"]
        )

        rag_chain = prompt | llm | StrOutputParser()
        for script in os.listdir("transcripts"):
            answer = rag_chain.invoke({script})
            print(answer)

    def add_documents(self):
        '''This function will be called in main and will handle the logic of what to do with provided docs'''
        pass


class OllamaRAG:
    def __init__(self, docs_list: list):
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
    doc_processor = DocumentProcessor()
    #doc_processor.pdf_processor(r"C:\Users\nwche\Dev\Python\Ollama-RAG\test_docs\table.pdf")
    doc_processor.youtube_processor("https://www.youtube.com/watch?v=KaLzkazrGcU&list=PLAVNxShcAmGHcoAJRbV0z-gzixdvEidxy&index=1")

    # # Init
    # document = TextLoader(r"test_docs/1. Procedural Introduction to Cybersecurity Litigation.txt").load()
    # q = "When was the mona lisa painted?"
    # rag_agent = OllamaRAG(docs_list=document)

    # # Grade document retrieval
    # grade = rag_agent.grade_retrieved_documents(question=q)
    
    # # Generate response
    # if grade == {'score': 'yes'}:
    #     rag_agent.generate_response(question=q)
