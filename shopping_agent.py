"""
shopping_agent.py

AI Shopping Assistant for Torob.

Provides:
- Pydantic models for input/output
- An LLM Agent configured with a schema-aware system prompt
- Helpers to parse LLM output and to fetch base/member random keys from Postgres
- A single entrypoint: `handle_shopping_instruction(instruction: str)` -> (ShoppingResponse, raw_output)

Requirements:
- python-dotenv (for loading .env)
- pydantic-ai (your installed version; module auto-handles provider vs configure)
- psycopg2
- httpx (if using provider)
- pydantic
"""

import os
from typing import List, Optional, Tuple, Dict, Any
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from pydantic_ai import Agent, UsageLimits, ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from sql.sql_agent import generate_sql_query
from sql.sql_utils import execute_sql, build_like_query_and_execute, build_exact_query_and_execute
from prompt.prompts import *
from utils import preprocess_persian


# load environment
load_dotenv()

# DB config from environment (ensure these are in your .env)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", ""),
    "port": int(os.getenv("DB_PORT", 0)),
    "dbname": os.getenv("DB_NAME", ""),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
}

INITIAL_THOUGHT_MODEL = os.getenv("INITIAL_THOUGHT_MODEL")
SHOPPING_MODEL = os.getenv("SHOPPING_MODEL")
OPENAI_API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")  # e.g. https://turbo.torob.com/v1


shopping_client = OpenAIChatModel(
    SHOPPING_MODEL,
    provider=OpenAIProvider(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        http_client=httpx.AsyncClient()
    ),
    settings=ModelSettings(temperature=0.001, max_tokens=1024)
)

initial_client = OpenAIChatModel(
    INITIAL_THOUGHT_MODEL,
    provider=OpenAIProvider(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        http_client=httpx.AsyncClient()
    ),
    settings=ModelSettings(temperature=0.001, max_tokens=1024)
)

cot_agent = Agent(
    name="TorobAssistantPlanner",
    model=initial_client,
    system_prompt=system_prompt.format(system_role = system_role_initial,
                                        tools = tools_info, 
                                       rules = rules_initial),
)

class ShoppingResponse(BaseModel):
    """
    The assistant's structured response.

    - message: Explanation or natural-language answer (or null).
    - base_random_keys: list of base product random_key values (or null).
    - member_random_keys: list of member random_key values (or null).
    """
    message: Optional[str] = None
    base_random_keys: Optional[List[str]] = None
    member_random_keys: Optional[List[str]] = None

shopping_agent = Agent(
    name="TorobShoppingAssistant",
    model=shopping_client,
    system_prompt=system_prompt.format(system_role = system_role,
                                        tools = tools_info, 
                                       rules = instructions_generated),
    tools=[build_exact_query_and_execute,
           build_like_query_and_execute, 
           generate_sql_query,
           execute_sql,],
    output_type=ShoppingResponse,  # Pydantic handles validation/parsing automatically
)

usage_limits = UsageLimits(request_limit=10, tool_calls_limit=10, output_tokens_limit=4096)

async def run_shopping_agent(
    instruction: str
) -> Tuple[ShoppingResponse, str]:
    """
    Run the Torob Shopping Assistant on a user instruction.

    Returns a validated ShoppingResponse and raw model output.
    """
    try:
        preprocessed_instruction = preprocess_persian(instruction)
        plan_output = await cot_agent.run(preprocessed_instruction)
        print(plan_output.output)
        prompt = "Input: " + preprocessed_instruction + "\n\n" + plan_output.output
        result = await shopping_agent.run(prompt,
                                usage_limits = usage_limits,
                                )
        return result
    except Exception as e:
        return ShoppingResponse(
                message=f"-- ERROR: {str(e)}",
                base_random_keys=None,
                member_random_keys=None)