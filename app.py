import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
import os
#import tempfile
import utils as utils
import importlib
importlib.reload(utils)

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")
st.set_page_config(layout="wide")


def main():
    '''
    Main function for the Streamlit-based ChatBot application.

    This function sets up and runs the ChatBot application, which allows users to upload and process
    multiple documents, create embeddings, and interact with a conversational ChatBot.

    Returns:
        None
    '''

    # Initialize session state
    utils.initializeSessionState()
    st.title("ChatBot de documentos usando llama2 :books:")   
   
    # Create embeddings  
    embeddings = SentenceTransformerEmbeddings(model_name="./model/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    # Create vector store
    vectorStore =  Chroma(persist_directory=DB_DIR, embedding_function = embeddings)
    
    # Create the chain object
    chain = utils.createConversationalChain(vectorStore)

    utils.displayChatHistory(chain)
    
        
if __name__ == "__main__":
    main()
    
#streamlit run app.py