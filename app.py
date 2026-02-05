from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer

app = FastAPI(title="LPU Admission Bot API")

class ChatRequest(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    return {"answer": answer(req.question)}
