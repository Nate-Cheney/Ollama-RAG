from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import gradio_app

def grade_retrieved_documents(question: str):
    # LLM
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
    documents = retriever.invoke(question)

    print(retrieval_grader.invoke({"question": question, "documents": documents}))