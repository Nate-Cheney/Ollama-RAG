def load_file(file_path=None):
    from os import environ    
    environ['USER_AGENT'] = 'RAG_TEST_AGENT'

    from langchain_community.document_loaders import TextLoader
    import easygui

    if not file_path:
        file_path = fr"{easygui.fileopenbox()}"
    
    docs = [TextLoader(file_path).load()]
    docs_list = [item for sublist in docs for item in sublist]  # flattens the list of lists (docs) into a single list (docs_list)

    return docs_list


def load_directory(dir_path=None):
    from os import listdir, environ    
    environ['USER_AGENT'] = 'RAG_TEST_AGENT'

    from langchain_community.document_loaders import TextLoader
    import easygui

    if not dir_path:
        dir_path = fr"{easygui.diropenbox()}"

    docs = [TextLoader(dir_path + "\\" + file).load() for file in listdir(dir_path)]
    docs_list = [item for sublist in docs for item in sublist]  # flattens the list of lists (docs) into a single list (docs_list)

    return docs_list


def load_websites(urls=None):
    from os import environ
    environ['USER_AGENT'] = 'RAG_TEST_AGENT'

    from langchain_community.document_loaders import WebBaseLoader


    if urls == None:
        urls = list(input("\nEnter the url(s) you'd like to scrape. If there's more than one enter them as follows:\
                          \n\thttps://example.com, https://google.com, etc.\n"))
        if urls == "":
            quit

    # Load documents from the urls list
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]  # flattens the list of lists (docs) into a single list (docs_list)

    return docs_list