from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
import os 


from langchain.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)


ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")


# Create vector database
def create_vector_database():
    
    """
    Creates a vector database using document loaders and embeddings.

    This function loads data from PDF, markdown and text files in the 'data/' directory,
    splits the loaded documents into chunks, transforms them into embeddings,
    and finally persists the embeddings into a Chroma vector database.

    """
    
    # Initialize loaders for different file types
    pdf_loader = DirectoryLoader("data/", glob="**/*.pdf", loader_cls=PyPDFLoader)
    markdown_loader = DirectoryLoader(
        "data/", glob="**/*.md", loader_cls=UnstructuredMarkdownLoader
    )
    text_loader = DirectoryLoader("data/", glob="**/*.txt", loader_cls=TextLoader)

    all_loaders = [pdf_loader, markdown_loader, text_loader]

    # Load documents from all loaders
    documents = []
    for loader in all_loaders:
        documents.extend(loader.load())
    
    # Split loaded documents into chunks
    print("splitting into chunks")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100, length_function=len)
    chunked_documents = text_splitter.split_documents(documents)
    
    #Create embeddings here
    print("Loading sentence transformers model")
    embeddings = SentenceTransformerEmbeddings(
                                model_name="./model/all-MiniLM-L6-v2",
                                model_kwargs={"device": "cpu"})
    
    # Create and persist a Chroma vector database from the chunked documents
    print(f"Creating embeddings. May take some minutes...")
    vector_database = Chroma.from_documents(
                                documents=chunked_documents, 
                                embedding=embeddings, 
                                persist_directory=DB_DIR)
    vector_database.persist()
    vector_database=None 

    print(f"Ingestion complete! You can now run to query your documents")

if __name__ == "__main__":
    create_vector_database()