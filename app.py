# app.py
import os
import re
from typing import List, Optional, Literal
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from shopping_agent import run_shopping_agent

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.environ['API_KEY']
BASE_URL = os.environ['BASE_URL']
MODEL_NAME = "gpt-4.1-mini"

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

app = FastAPI()

# ------ Pydantic models ------
class Message(BaseModel):
    type: Literal["text", "image"]
    content: str

class ChatRequest(BaseModel):
    chat_id: str
    messages: List[Message]

class ChatResponse(BaseModel):
    message: Optional[str] = None
    base_random_keys: Optional[List[str]] = None
    member_random_keys: Optional[List[str]] = None

# ------ Patterns ------
RE_BASE = re.compile(r"return base random key:\s*([A-Za-z0-9\-_:]+)", re.IGNORECASE)
RE_MEMBER = re.compile(r"return member random key:\s*([A-Za-z0-9\-_:]+)", re.IGNORECASE)

# ------ Endpoint ------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):

    print(dict(req))

    # very small defensive check
    if not req.messages:
        resp = ChatResponse(message=None, base_random_keys=None, member_random_keys=None)
        return dict(resp)

    last = req.messages[-1]
    content = last.content.strip()

    # 1) ping
    if last.type == "text" and content == "ping":
        resp = ChatResponse(message="pong", base_random_keys=None, member_random_keys=None)
        return dict(resp)

    # 2) return base random key
    m_base = RE_BASE.search(content)
    if m_base:
        key = m_base.group(1)
        resp = ChatResponse(message=None, base_random_keys=[key], member_random_keys=None)
        return dict(resp)

    # 3) return member random key
    m_member = RE_MEMBER.search(content)
    if m_member:
        key = m_member.group(1)
        resp = ChatResponse(message=None, base_random_keys=None, member_random_keys=[key])
        return resp
    

    _, output_dict = await run_shopping_agent(content)

    print(output_dict)

    return output_dict
