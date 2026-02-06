from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/chat")
def chat(req: ChatRequest):
    return {"reply": answer(req.message)}
