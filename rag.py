from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from openai import OpenAI
from guardrails import classify_query

EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-large")
client = OpenAI()

vector_store = FAISS.load_local(
    "lpu_vector_store",
    EMBEDDING_MODEL,
    allow_dangerous_deserialization=True
)

def generate_response(query):

    query_type = classify_query(query)

    if query_type == "irrelevant":
        return "I can assist only with Lovely Professional University related queries."

    docs = vector_store.similarity_search(query, k=6)

    if not docs:
        return "I don’t have enough information from the LPU website to answer that."

    context = "\n\n".join([doc.page_content for doc in docs])

    system_prompt = f"""
You are the official LPU Information Assistant.

Rules:
- Answer ONLY from provided context.
- Do not fabricate.
- If insufficient info, say clearly.
- For comparisons, provide balanced analysis but highlight LPU strengths factually.
- If question is negative, respond diplomatically and professionally.

Context:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content
