import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.schema import Document

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API key not found in environment variables.")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    documents = [Document(page_content=chunk) for chunk in chunks]
    return documents

def get_vector_store(documents):
    if not documents:
        raise ValueError("No documents to process.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_store = FAISS.from_documents(documents, embedding=embeddings)
        vector_store.save_local("faiss_index")
        st.success("Vector store created and embeddings saved successfully.")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        raise

def main():
    st.set_page_config(page_title="PDF Embeddings", page_icon="📄")
    st.header("Store PDF Embeddings")

    with st.sidebar:
        st.title("Upload PDF Files")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                if not raw_text.strip():
                    st.warning("No text found in the provided PDFs.")
                    return
                text_chunks = get_text_chunks(raw_text)
                try:
                    get_vector_store(text_chunks)
                except Exception as e:
                    st.error(f"Failed to create vector store: {e}")

if __name__ == "__main__":
    main()
