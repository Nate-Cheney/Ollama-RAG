def create_promt():
    '''
    The create_prompt() function isn't meant to be called directly.
    It gets called within the inititalize_[model] function
    '''
    from langchain.prompts import PromptTemplate

    prompt_dict = {  # All predefined prompts
        "Prompt 1": 
        """
        You are an assistant for question-answering tasks.
        Use the following documents to answer the question.
        If you don't know the answer, just say that you don't know.
        Use three sentences maximum and keep the answer concise:
        """,
    
        "Custom":
        """
        If you'd like to use a custom prompt press enter.
        """
    }

    print("Prompt Options:\n")
    for key in prompt_dict:
        print(f"{key} \n{prompt_dict[key]}")
    prompt_selection = input(f"\nSelect one of the prompt options above. Enter the number that corresponds with the desired prompt.\nFor example: 1\n")

    if prompt_selection == "":
        prompt_template = input("Enter a prompt:\n")
    elif prompt_selection == "1":
        prompt_template = prompt_dict["Prompt 1"]
    else:
        print("You entered an invalid option.\n")
        create_promt()

    prompt_template += """
        Question: {question}
        Documents: {documents}
        Answer:
        """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["question", "documents"]
    )

    return prompt


def initialize_llama31():
    from langchain_ollama import ChatOllama
    from langchain_core.output_parsers import StrOutputParser

    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
    )
    prompt = create_promt()
    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain


# Define the RAG application class
class RAGApplication:
    def __init__(self, retriever):
        self.retriever = retriever
        self.rag_chain = initialize_llama31()

    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer


class RAGInterface:
    def __init__(self):
        import preprocessing
        import documents

        # Gather documents
        q = "Enter the number that corresponds with type of documents you'd like to process. \
        \n\t1 - Files from a directory \
        \n\t2 - Information from a website"
        to_load = list(input(q))
        for item in to_load:
            if item == "1":
                docs_list = documents.load_directory()
            elif item == "2":
                docs_list = documents.load_website()

        # Chunk and embed documents
        retriever = preprocessing.embed_doc_splits(docs_list)

        # Initialize the RAG application
        self.rag_application = RAGApplication(retriever)

    def ask_question(self, continue_asking="y"):
        while continue_asking.lower() == "y":
            question = input("Ask a question: ")
            answer = self.rag_application.run(question)
            print("Answer:\n", answer)

            continue_asking = input(f"\nWould you like to ask another question? (y/n)")

    
if __name__ == "__main__":
    rag_interface = RAGInterface()
    rag_interface.ask_question()
