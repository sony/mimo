from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import  AzureOpenAIEmbeddings


    
def customized_trend_retriever(reflect_loader, embedding_api_key, openai_embedding_azure_openai_endpoint):
    
    """
    Read the input file and domain knowledge file, and get the embeddings.

    Parameters:
    - input_file_loader: A loader object for the input file.
    - rule_loader: A loader object for the domain knowledge file.
    - embedding_api_key: The API key for the embedding service.

    Returns:
    - embedded_input: The embeddings of the input file.
    - embedded_rule: The embeddings of the domain knowledge file.
    """
        
    docs = reflect_loader.load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size=100000, chunk_overlap=20000
    ).split_documents(docs)
    reflection_vector = FAISS.from_documents(documents, AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002", openai_api_version="2023-05-15", openai_api_key=embedding_api_key, azure_endpoint=openai_embedding_azure_openai_endpoint))
    reflection_file_retriever = reflection_vector.as_retriever(search_kwargs={"k": 10000})

    return reflection_file_retriever

def customized_retriever(input_file_loader, rule_loader, embedding_api_key, openai_embedding_azure_openai_endpoint):
    
    """
    Read the input file and domain knowledge file, and get the embeddings.

    Parameters:
    - input_file_loader: A loader object for the input file.
    - rule_loader: A loader object for the domain knowledge file.
    - embedding_api_key: The API key for the embedding service.

    Returns:
    - embedded_input: The embeddings of the input file.
    - embedded_rule: The embeddings of the domain knowledge file.
    """
        
    docs = input_file_loader.load()
    documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    input_vector = FAISS.from_documents(documents, AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002", openai_api_version="2023-05-15", openai_api_key=embedding_api_key, azure_endpoint=openai_embedding_azure_openai_endpoint))
    input_file_retriever = input_vector.as_retriever(search_kwargs={"k": 1000})

    rule_docs = rule_loader.load()
    rule_documents = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(rule_docs)
    rule_vector = FAISS.from_documents(rule_documents, AzureOpenAIEmbeddings(azure_deployment="text-embedding-ada-002", openai_api_version="2023-05-15", openai_api_key=embedding_api_key, azure_endpoint=openai_embedding_azure_openai_endpoint))
    rule_retriever = rule_vector.as_retriever(search_kwargs={"k": 1000})
    
    return input_file_retriever, rule_retriever
    