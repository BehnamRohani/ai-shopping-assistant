# app.py
import os
import re
import json
import psycopg2
import psycopg2.extras
from typing import List, Optional, Literal
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi import Request
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

# ------ Database Helpers ------
def get_db_conn():
    return psycopg2.connect(**DB_CONFIG)

def init_logs_table():
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id SERIAL PRIMARY KEY,
            time TIMESTAMP NOT NULL,
            input JSONB,
            output JSONB
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def insert_log(input_data: dict, output_data: dict):
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO logs (time, input, output) VALUES (%s, %s, %s)",
            (datetime.utcnow(), json.dumps(input_data), json.dumps(output_data))
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"[ERROR] Failed to insert log: {e}")

# ------ Lifespan Context ------
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_logs_table()
    yield
    # Shutdown (if needed)
    # e.g., close connections

app = FastAPI(lifespan=lifespan)

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
    try:
        input_dict = req.model_dump()
        print("[INPUT]", input_dict)

        # very small defensive check
        if not req.messages:
            resp = ChatResponse()
            insert_log(input_dict, resp.model_dump())
            return resp

        last = req.messages[-1]
        content = last.content.strip()

        # 1) ping
        if last.type == "text" and content == "ping":
            resp = ChatResponse(message="pong")
            insert_log(input_dict, resp.model_dump())
            return resp

        # 2) return base random key
        m_base = RE_BASE.search(content)
        if m_base:
            key = m_base.group(1)
            resp = ChatResponse(base_random_keys=[key])
            insert_log(input_dict, resp.model_dump())
            return resp

        # 3) return member random key
        m_member = RE_MEMBER.search(content)
        if m_member:
            key = m_member.group(1)
            resp = ChatResponse(member_random_keys=[key])
            insert_log(input_dict, resp.model_dump())
            return resp

        # 4) shopping agent
        _, output_dict = await run_shopping_agent(instruction=content, use_initial_plan=True)
        print("[OUTPUT]", output_dict)

        insert_log(input_dict, output_dict)
        return output_dict

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
