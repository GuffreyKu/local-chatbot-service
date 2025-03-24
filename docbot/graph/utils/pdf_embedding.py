from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

def multi_pdf_embedding(pdf_paths):
    all_documents = []
    # Loop through each PDF path and load the documents
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()  # Load returns a list of Document objects
        all_documents.extend(documents)

    # Split the combined PDF content into smaller text chunks for better embedding results
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(all_documents)
    # Initialize the embedding model
    embedding_model = OllamaEmbeddings(
        model="bge-m3",
    )
    # print(docs)
    # Create a vector store (using FAISS) from the split documents and their embeddings
    vector_store = FAISS.from_documents(docs, embedding_model)
    
    # print("Total vectors stored:", vector_store.index.ntotal)
    # print("Dimension of each vector:", vector_store.index.d)
    return vector_store

if __name__ == '__main__':
    import os
    pdf_list = ["../../../data/pdf/"+pdf_path for pdf_path in os.listdir("../../../data/pdf/")]
    vector_store = multi_pdf_embedding(pdf_list)
    print("Top K : ", vector_store.similarity_search("交通費的資訊？"))