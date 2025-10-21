# import basics
import os
import time
from dotenv import load_dotenv

# vector store
from langchain_community.vectorstores import FAISS

# import langchain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

#documents
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv() 

# initialize embeddings (Gemini)
embeddings = GoogleGenerativeAIEmbeddings(
    model=os.environ.get("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
    google_api_key=os.environ.get("GOOGLE_API_KEY"),
    transport="rest",
)


# loading the PDF document
loader = PyPDFDirectoryLoader("documents/")

raw_documents = loader.load()

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=400,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
documents = text_splitter.split_documents(raw_documents)

# build FAISS index and persist
index_dir = "faiss_gemini"
vector_store = FAISS.from_documents(documents=documents, embedding=embeddings)
vector_store.save_local(index_dir)
print(f"Ingestion complete. FAISS index saved at: {index_dir}")