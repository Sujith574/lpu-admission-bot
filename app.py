from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag import answer

app = FastAPI()

# ✅ CORS — REQUIRED
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for now, lock later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    reply = answer(req.message)
    return {"reply": reply}
