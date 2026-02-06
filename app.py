from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer
import asyncio

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

@app.get("/")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(req: ChatRequest):
    reply = await asyncio.to_thread(answer, req.message)
    return {"reply": reply}
