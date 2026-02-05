import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

with open("data/lpu_pages.json", "r", encoding="utf-8") as f:
    pages = json.load(f)

texts = [p["content"] for p in pages]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80
)

docs = splitter.create_documents(texts)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embeddings)
db.save_local("vectorstore")

print("Vector store created")
