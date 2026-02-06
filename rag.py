import os
from google.cloud import storage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from guardrails import classify_query

# ---------------- HARD FACTS ----------------
LPU_FACTS = {
    "location": "Lovely Professional University is located in Phagwara, Punjab, India near NH-44.",
    "address": "Lovely Professional University, Jalandhar–Delhi G.T. Road, Phagwara, Punjab 144411, India.",
    "naac": "LPU is accredited by NAAC with A++ grade (CGPA 3.68).",
    "nirf": "LPU is ranked 31st in India by NIRF.",
    "pro_chancellor": "Dr. Ashok Kumar Mittal is the Pro-Chancellor of LPU."
}

# ---------------- GEMINI ----------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "models/gemini-2.5-flash"

# ---------------- VECTORSTORE ----------------
def load_vectorstore():
    if os.path.exists("vectorstore/index.faiss"):
        return

    storage_client = storage.Client()
    bucket = storage_client.bucket("lpu-admission-bot-data")
    blobs = storage_client.list_blobs(bucket, prefix="vectorstore/")

    os.makedirs("vectorstore", exist_ok=True)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        path = blob.name.replace("vectorstore/", "")
        dest = os.path.join("vectorstore", path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)

load_vectorstore()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

SYSTEM_PROMPT = """
You are an official admission assistant of Lovely Professional University (LPU).
Be clear, professional, and LPU-focused.
"""

def answer(query: str) -> str:
    q = query.lower()

    # ---- Identity ----
    if "who" in q and "you" in q:
        return (
            "I’m a virtual admission assistant built to help with "
            "Lovely Professional University admissions and information."
        )

    # ---- Hard facts ----
    for key in LPU_FACTS:
        if key.replace("_", " ") in q:
            return LPU_FACTS[key]

    if "fee" in q:
        return (
            "Fees vary by program and scholarships through LPUNEST. "
            "Please check the official LPU admission portal for exact details."
        )

    intent = classify_query(query)
    if intent == "negative":
        return "LPU continuously improves academics, placements, and infrastructure."

    if intent == "unrelated":
        return "Please ask about LPU admissions, courses, or campus life."

    docs = db.similarity_search(query, k=6)
    context = "\n".join(d.page_content for d in docs)

    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion:\n{query}"

    try:
        res = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return res.text
    except Exception:
        return "I’m having trouble answering right now. Please try again."
