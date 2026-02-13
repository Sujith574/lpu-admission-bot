from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

import json
import os

EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-large")

def load_scraped_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_vector_store(data):
    texts = []
    metadatas = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120
    )

    for page in data:
        chunks = splitter.split_text(page["content"])

        for chunk in chunks:
            texts.append(chunk)
            metadatas.append({
                "url": page["url"]
            })

    vector_store = FAISS.from_texts(
        texts,
        embedding=EMBEDDING_MODEL,
        metadatas=metadatas
    )

    vector_store.save_local("lpu_vector_store")

if __name__ == "__main__":
    data = load_scraped_data("scraped_data.json")
    create_vector_store(data)

