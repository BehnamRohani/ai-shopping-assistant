# app.py
import os
import re
from typing import List, Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi import Request
from pydantic import BaseModel
from dotenv import load_dotenv
from shopping_agent_old import run_shopping_agent
from sql.similarity_search_db import similarity_search
from sql.sql_utils import init_logs_table, insert_log, insert_chat, get_chat_history
from agents.torob_agents import TorobHybridAgent
from pydantic_ai import UsageLimits

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.environ['API_KEY']
BASE_URL = os.environ['BASE_URL']
MODEL_NAME = "gpt-4.1-mini"

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

usage_limits = UsageLimits(request_limit=30, tool_calls_limit=30, output_tokens_limit=4096)

myagent = TorobHybridAgent()

# ------ Endpoint ------
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        input_dict = req.model_dump()
        all_texts = [m["content"] for m in input_dict["messages"] if m["type"] == "text"]
        print("[INPUT]", all_texts[0] + "\n" + input_dict['chat_id'])

        last = req.messages[-1]
        content = last.content.strip()

        # Similarity Search DB
        if input_dict['chat_id'] == 'retrieve_similar':
            results = similarity_search(content, top_k = 5, probes=10)
            rks = [res[0] for res in results]
            names = [res[1] for res in results]
            similarities = [f"{res[2]:.4f}" for res in results]
            message_list = [(names[i] + " -> " + similarities[i]) for i in range(len(names))]
            resp = ChatResponse(message = "\n".join(message_list), base_random_keys=rks)
            return resp

        # very small defensive check
        if not req.messages:
            resp = ChatResponse()
            # insert_log(input_dict, resp.model_dump())
            return resp

        # 1) ping
        if last.type == "text" and content == "ping":
            resp = ChatResponse(message="pong")
            # insert_log(input_dict, resp.model_dump())
            return resp

        # 2) return base random key
        m_base = RE_BASE.search(content)
        if m_base:
            key = m_base.group(1)
            resp = ChatResponse(base_random_keys=[key])
            # insert_log(input_dict, resp.model_dump())
            return resp

        # 3) return member random key
        m_member = RE_MEMBER.search(content)
        if m_member:
            key = m_member.group(1)
            resp = ChatResponse(member_random_keys=[key])
            # insert_log(input_dict, resp.model_dump())
            return resp

        # 4) shopping agent
        # result, output_dict = await run_shopping_agent(input_dict=input_dict, 
        #                                           use_initial_plan=True,
        #                                           use_parser_output=True,
        #                                           use_initial_similarity_search=True)

        result, output_dict = await myagent.run(input_dict=input_dict,
                                                usage_limits=usage_limits,
                                                use_initial_similarity_search=True)
        print("[OUTPUT]", output_dict)
        extra_info = output_dict.pop("extra_info", None)  # remove from output_dict
        insert_chat(input_dict, output_dict, extra_info=extra_info)
        # print(result.all_messages())
        insert_log(input_dict, output_dict)
        # Remove `finished` from the output dict before returning
        output_dict.pop("finished", None)
        return output_dict

    except Exception as e:
        print(f"[ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))
