# app.py
import os
import re
import json
import uuid
import time
from typing import List, Optional, Literal
from datetime import datetime
from fastapi import FastAPI, Request, Header, HTTPException
from pydantic import BaseModel

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
    start = time.time()
    # optional auth

    # very small defensive check
    if not req.messages:
        resp = ChatResponse(message=None, base_random_keys=None, member_random_keys=None)
        return resp

    last = req.messages[-1]
    content = last.content.strip()

    # 1) ping
    if last.type == "text" and content == "ping":
        resp = ChatResponse(message="pong", base_random_keys=None, member_random_keys=None)
        return resp

    # 2) return base random key
    m_base = RE_BASE.search(content)
    if m_base:
        key = m_base.group(1)
        resp = ChatResponse(message=None, base_random_keys=[key], member_random_keys=None)
        return resp

    # 3) return member random key
    m_member = RE_MEMBER.search(content)
    if m_member:
        key = m_member.group(1)
        resp = ChatResponse(message=None, base_random_keys=None, member_random_keys=[key])
        return resp

    # default fallback
    resp = ChatResponse(message="متأسفم، متوجه نشدم. لطفاً دقیق‌تر بپرسید.", base_random_keys=None, member_random_keys=None)
    return resp
