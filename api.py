import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from chatbot import ChatSession


app = FastAPI(title="NHS 101 API", version="0.1.0")
sessions = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    user_profile: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session = sessions.get(request.session_id)
    session_id = request.session_id or uuid.uuid4().hex

    if session is None:
        session = ChatSession()
        sessions[session_id] = session

    result = session.send_message(message)

    return ChatResponse(session_id=session_id, reply=result["reply"], user_profile=result["user_profile"])
