import os

from google.cloud import storage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from google import genai
from guardrails import classify_query

# =====================================================
# HARD FACTS (ONLY STABLE FACTS)
# =====================================================
LPU_FACTS = {
    "location": (
        "Lovely Professional University is located in Phagwara, Punjab, India, "
        "near Jalandhar on National Highway 44 (NH-44)."
    ),
    "address": (
        "Lovely Professional University, Jalandhar–Delhi G.T. Road, "
        "Phagwara, Punjab 144411, India."
    ),
    "naac": (
        "Lovely Professional University is accredited by NAAC with an A++ grade "
        "with a CGPA of 3.68, which is the highest grade awarded in a first accreditation cycle."
    ),
    "nirf": (
        "Lovely Professional University is ranked 31st among both government and "
        "private universities in India by the National Institutional Ranking Framework (NIRF)."
    ),
    "pro_chancellor": (
        "Dr. Ashok Kumar Mittal is the Pro-Chancellor of Lovely Professional University."
    ),
}

# =====================================================
# GEMINI CLIENT
# =====================================================
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL_NAME = "models/gemini-2.5-flash"

# =====================================================
# VECTOR STORE DOWNLOAD (CLOUD RUN SAFE)
# =====================================================
def download_vectorstore():
    client = storage.Client()
    bucket = client.bucket("lpu-admission-bot-data")

    blobs = client.list_blobs(bucket, prefix="vectorstore/")

    os.makedirs("vectorstore", exist_ok=True)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue

        local_path = blob.name.replace("vectorstore/", "")
        local_file = os.path.join("vectorstore", local_path)

        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        blob.download_to_filename(local_file)

# Download only once
if not os.path.exists("vectorstore/index.faiss"):
    download_vectorstore()

# =====================================================
# LOAD VECTOR STORE
# =====================================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# =====================================================
# SYSTEM PROMPT
# =====================================================
SYSTEM_PROMPT = """
You are an official admission assistant of Lovely Professional University (LPU).

Rules:
- Maintain a positive, professional, and helpful tone.
- Focus on assisting users with LPU-related information.
- Never promote or recommend other universities.
"""

# =====================================================
# MAIN ANSWER FUNCTION
# =====================================================
def answer(query: str) -> str:
    q = query.lower().strip()

    # ---------------- META QUESTIONS ----------------
    if any(x in q for x in [
        "who developed you",
        "who created you",
        "who made you",
        "are you ai",
        "are you human",
        "what are you"
    ]):
        return (
            "I’m a virtual admission assistant created to help students and parents "
            "with information related to Lovely Professional University. "
            "My role is to guide you with admissions, programs, campus facilities, "
            "and other LPU-related queries."
        )

    # ---------------- HARD FACT OVERRIDES ----------------
    if "where is lpu" in q or "location of lpu" in q:
        return LPU_FACTS["location"]

    if "address" in q and "lpu" in q:
        return LPU_FACTS["address"]

    if "naac" in q:
        return LPU_FACTS["naac"]

    if "nirf" in q or "ranking" in q:
        return LPU_FACTS["nirf"]

    if "pro chancellor" in q or "pro-chancellor" in q:
        return LPU_FACTS["pro_chancellor"]

    # ---------------- FEES UX ----------------
    if "fee" in q or "fees" in q:
        return (
            "The fee structure at Lovely Professional University varies depending on the "
            "program and scholarship eligibility through LPUNEST.\n\n"
            "For the most accurate and updated fee details, "
            "I recommend checking the official LPU admission portal or applying for LPUNEST."
        )

    # ---------------- INTENT ----------------
    intent = classify_query(query)

    if intent == "negative":
        return (
            "I understand your concern. Lovely Professional University continuously "
            "works to improve academic quality, infrastructure, industry collaboration, "
            "and student support services.\n\n"
            "If you have a specific concern or need information about programs or admissions, "
            "I’ll be glad to assist."
        )

    if intent == "unrelated":
        return (
            "I’m here to help with queries related to Lovely Professional University. "
            "Please feel free to ask about admissions, courses, campus life, or facilities."
        )

    # ---------------- RAG ----------------
    docs = db.similarity_search(query, k=8)
    context = "\n".join(d.page_content for d in docs)

    prompt = f"""
{SYSTEM_PROMPT}

Context:
{context}

User Question:
{query}
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
        return response.text

    except Exception:
        if any(x in q for x in ["better", "compare", "vs"]):
            return (
                "When comparing universities, students usually consider academic quality, "
                "accreditation, infrastructure, placements, and overall campus experience.\n\n"
                "Lovely Professional University stands out due to its A++ NAAC accreditation, "
                "strong NIRF ranking, global recognition, and industry-focused education.\n\n"
                "Overall, LPU offers a future-ready and well-rounded learning environment."
            )

        return (
            "I’m currently unable to retrieve detailed information, but I’ll be happy "
            "to assist you with any LPU-related queries."
        )
