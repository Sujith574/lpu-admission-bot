import os
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google.cloud import storage
from google import genai
from guardrails import classify_query

# ================= HARD FACTS =================
LPU_FACTS = {
    "location": "Lovely Professional University is located in Phagwara, Punjab, India near NH-44.",
    "address": "Lovely Professional University, Jalandhar–Delhi G.T. Road, Phagwara, Punjab 144411, India.",
    "naac": "LPU is NAAC accredited with A++ grade (CGPA 3.68).",
    "nirf": "LPU is ranked among the top universities in India by NIRF.",
    "pro_chancellor": "Dr. Ashok Kumar Mittal is the Pro-Chancellor of LPU."
}

MODEL_NAME = "models/gemini-2.5-flash"

# ================= LAZY GLOBALS =================
_db: Optional[FAISS] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None
_client: Optional[genai.Client] = None


# ================= INIT FUNCTIONS =================
def _init_gemini():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _download_vectorstore():
    if os.path.exists("vectorstore/index.faiss"):
        return

    client = storage.Client()
    bucket = client.bucket("lpu-admission-bot-data")

    os.makedirs("vectorstore", exist_ok=True)

    for blob in client.list_blobs(bucket, prefix="vectorstore/"):
        if blob.name.endswith("/"):
            continue
        path = blob.name.replace("vectorstore/", "")
        local = os.path.join("vectorstore", path)
        os.makedirs(os.path.dirname(local), exist_ok=True)
        blob.download_to_filename(local)


def _init_vectorstore():
    global _db, _embeddings
    if _db is not None:
        return

    _download_vectorstore()

    _embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    _db = FAISS.load_local(
        "vectorstore",
        _embeddings,
        allow_dangerous_deserialization=True
    )


# ================= MAIN ANSWER =================
def answer(query: str) -> str:
    q = query.lower().strip()

    # Hard facts
    for key in LPU_FACTS:
        if key.replace("_", " ") in q:
            return LPU_FACTS[key]

    intent = classify_query(query)

    if intent == "negative":
        return (
            "Lovely Professional University continuously improves academic quality, "
            "infrastructure, and student outcomes. If you have a specific concern, I can help."
        )

    if intent == "unrelated":
        return "I can help with admissions, courses, rankings, campus life, and facilities at LPU."

    # Lazy init (CRITICAL)
    _init_gemini()
    _init_vectorstore()

    docs = _db.similarity_search(query, k=6)
    context = "\n".join(d.page_content for d in docs)

    prompt = f"""
You are an official LPU admission assistant.

Context:
{context}

Question:
{query}
"""

    response = _client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text
