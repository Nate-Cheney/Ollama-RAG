ollama_rag_ascii_art = '''
   ____  _ _                         _____            _____ 
  / __ \| | |                       |  __ \     /\   / ____|
 | |  | | | | __ _ _ __ ___   __ _  | |__) |   /  \ | |  __ 
 | |  | | | |/ _` | '_ ` _ \ / _` | |  _  /   / /\ \| | |_ |
 | |__| | | | (_| | | | | | | (_| | | | \ \  / ____ \ |__| |
  \____/|_|_|\__,_|_| |_| |_|\__,_| |_|  \_\/_/    \_\_____|
'''


def create_promt():
    '''
    The create_prompt() function isn't meant to be called directly.
    It gets called within the inititalize_[model] function
    '''
    from langchain.prompts import PromptTemplate

    prompt_dict = {  # All predefined prompts
        "1": 
        """
        You are an assistant for question-answering tasks.
        Use the following documents to answer the question.
        If you don't know the answer, just say that you don't know.
        Use UP TO four sentences maximum and keep the answer concise:
        """,
        "Custom":
        """
        Press enter.
        """
    }

    print("\nPrompt Options:\n")
    for key in prompt_dict:
        print(f"{key} \n{prompt_dict[key]}")
    prompt_selection = input(f"Select one of the prompt options above.\n")

    if prompt_selection == "":
        prompt_template = input("Enter a prompt:\n")
    elif prompt_selection == "1":
        prompt_template = prompt_dict["1"]
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
        import time
        import utils.preprocessing as preprocessing
        import documents as documents


        print(ollama_rag_ascii_art)
        time.sleep(0.5)

        # Gather documents
        doc_question = "Select source of the documents you'd like to process. \n\t1 - File \n\t2 - Directory \n\t3 - Website \n"
        to_load = list(input(doc_question))
        docs_list = []
        for item in to_load:
            if item == "1":
                docs_list += documents.load_file()
            elif item == "2":
                docs_list += documents.load_directory()
            elif item == "3":
                docs_list += documents.load_websites()

        # Chunk and embed documents
        doc_splits = preprocessing.chunk_documents(docs_list)
        retriever = preprocessing.embed_doc_splits(doc_splits)

        # Initialize the RAG application
        self.rag_application = RAGApplication(retriever)


    def generate_response(self, continue_asking="y"):
        while continue_asking.lower() == "y":
            question = input("\nAsk a question: ")
            answer = self.rag_application.run(question)
            print("\nAnswer:\n",answer)

            continue_asking = input(f"\n\nWould you like to ask another question? (y/n) ")

    
if __name__ == "__main__":
    rag_interface = RAGInterface()
    rag_interface.generate_response()
    