import os
from google.cloud import storage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from guardrails import classify_query

# ------------------ GLOBALS (lazy loaded) ------------------
db = None
embeddings = None

# ------------------ FACTS ------------------
LPU_FACTS = {
    "location": "Lovely Professional University is located in Phagwara, Punjab, India.",
    "address": "Lovely Professional University, Phagwara, Punjab 144411, India.",
    "naac": "LPU is accredited by NAAC with an A++ grade.",
    "nirf": "LPU is ranked 31st in India by NIRF.",
    "pro_chancellor": "Dr. Ashok Kumar Mittal is the Pro-Chancellor of LPU.",
}

# ------------------ GEMINI ------------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "models/gemini-2.5-flash"

# ------------------ VECTORSTORE ------------------
def init_vectorstore():
    global db, embeddings
    if db is not None:
        return

    if not os.path.exists("vectorstore/index.faiss"):
        storage_client = storage.Client()
        bucket = storage_client.bucket("lpu-admission-bot-data")
        blobs = storage_client.list_blobs(bucket, prefix="vectorstore/")

        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            local_path = blob.name.replace("vectorstore/", "")
            local_file = os.path.join("vectorstore", local_path)
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            blob.download_to_filename(local_file)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

# ------------------ MAIN ------------------
def answer(query: str) -> str:
    init_vectorstore()  # 🔥 LAZY LOAD HERE

    q = query.lower()

    for key, value in LPU_FACTS.items():
        if key in q:
            return value

    intent = classify_query(query)
    if intent in ["negative", "unrelated"]:
        return "I can help only with LPU admission related queries."

    docs = db.similarity_search(query, k=5)
    context = "\n".join(d.page_content for d in docs)

    prompt = f"""
You are an official LPU admission assistant.

Context:
{context}

Question:
{query}
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )
    return response.text
