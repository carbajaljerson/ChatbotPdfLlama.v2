import streamlit as st 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
#from streamlit_chat import message
import os 

ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")
st.set_page_config(layout="wide")



@st.cache_resource
def qa_llm():
   
    llm = CTransformers(model="./model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                        streaming=True, 
                        callbacks=[StreamingStdOutCallbackHandler()],
                        model_type="llama", config={'max_new_tokens': 512, 'temperature': 0.001})
    
    
    embeddings = SentenceTransformerEmbeddings(model_name="./model/all-MiniLM-L6-v2",
                                               model_kwargs={'device': 'cpu'})
    db = Chroma(persist_directory=DB_DIR, embedding_function = embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    answer = None
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer


def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF 🦜📄 </h1>", unsafe_allow_html=True)

    with st.expander("About the App"):
        st.markdown("""
                    This is a Generative AI powered Question and Anwering app that responds question 
                    about your PDF file
                    """
        )
    question = st.text_area("Enter YourQuestion")
    if st.button("Search"):
        st.info("Your question: "+question )
        st.info("Your Answer")
        answer = process_answer(question)
        st.write(answer)
        
if __name__ == "__main__":
    main()
    
#streamlit run app.py