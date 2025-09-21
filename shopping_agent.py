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
from sql.similarity_search_db import similarity_search


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
OUTPUT_PARSER_MODEL = os.getenv("OUTPUT_PARSER_MODEL")

OPENAI_API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")  # e.g. https://turbo.torob.com/v1

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

shopping_client = OpenAIChatModel(
    SHOPPING_MODEL,
    provider=OpenAIProvider(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        http_client=httpx.AsyncClient()
    ),
    settings=ModelSettings(temperature=0.001, max_tokens=1024)
)

shopping_agent = Agent(
    name="TorobShoppingAssistant",
    model=shopping_client,
    system_prompt=system_prompt.format(system_role = system_role,
                                        tools = tools_info, 
                                       rules = instructions_generated),
    tools=[similarity_search,
           generate_sql_query,
           execute_sql,],
    output_type=ShoppingResponse,  # Pydantic handles validation/parsing automatically
)

parser_client = OpenAIChatModel(
    OUTPUT_PARSER_MODEL,
    provider=OpenAIProvider(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        http_client=httpx.AsyncClient()
    ),
    settings=ModelSettings(temperature=0.00001, max_tokens=100)
)

parser_agent = Agent(
    name="TorobParser",
    model=parser_client,
    system_prompt=parser_system_prompt,
)

usage_limits = UsageLimits(request_limit=30, tool_calls_limit=30, output_tokens_limit=4096)

from typing import Tuple, Optional

async def run_shopping_agent(
    instruction: str,
    use_initial_plan: bool = True,
    use_parser_output: bool = True,
    use_initial_similarity_search: bool = True,
) -> Tuple[Optional[ShoppingResponse], dict]:
    """
    Execute the Torob Shopping Assistant pipeline for a given user instruction.

    Args:
        instruction (str): User input.
        use_initial_plan (bool): If True, generate a high-level plan via CoT agent.
        use_parser_output (bool): If True, normalize final message using parser agent.
        use_initial_similarity_search (bool): If True, run similarity search first and include results in prompt.

    Steps:
    1. Preprocess the user instruction (Persian normalization).
    2. Optionally generate a high-level plan using the CoT agent.
    3. Optionally run similarity search and append candidates.
    4. Run the shopping agent with combined input, plan, and candidates.
    5. Optionally normalize the output message via the parser agent.

    Returns:
        Tuple[ShoppingResponse | None, dict]:
        - ShoppingResponse object with structured fields (message, base_random_keys, member_random_keys).
        - Dictionary representation of the output with normalized or raw message.
          Random keys remain as generated by the shopping agent.

    On error:
        - Returns None for ShoppingResponse
        - Dictionary with error message in 'message' and None for key lists.
    """
    try:
        # Step 1: preprocess input
        preprocessed_instruction = preprocess_persian(instruction)

        # Step 2: optionally generate plan
        plan_output = None
        plan_text = ""
        if use_initial_plan:
            plan_output = await cot_agent.run(preprocessed_instruction)
            plan_text = plan_output.output or ""
            print(plan_text)

        # Step 3: optionally run similarity search
        similarity_text = ""
        if use_initial_similarity_search:
            try:
                candidates = similarity_search(preprocessed_instruction, top_k=5, probes=20)
                if candidates[0][-1] > 0.7:
                    # candidates is expected to be list[tuple[str, str, float]]
                    rows = []
                    for rk, name, score in candidates:
                        rows.append(f"{rk} -> {name} -> similarity: {score:.4f}")
                    similarity_text = "\n".join(rows)
                    print("Similarity search results:\n", similarity_text)
            except Exception as e:
                print(f"Similarity search failed: {e}")

        # Step 4: build prompt for shopping agent
        prompt = "Input: " + preprocessed_instruction
        if plan_text:
            prompt += "\n\nPlan:\n" + plan_text
        if similarity_text:
            prompt += "\n\nInitial Similarity Search Candidates:\n" + similarity_text
            prompt += "\n" + "The initial similarity search results are provided for convenience, but you should also invoke similarity_search yourself whenever product identification is required."

        result = await shopping_agent.run(prompt, usage_limits=usage_limits)

        # Convert to dict for output formatting
        output_dict = dict(result.output)

        # Step 5: optionally normalize message
        message_out = result.output.message
        if message_out and use_parser_output:
            preprocessed_output = preprocess_persian(message_out)
            parser_input = parser_prompt.format(
                input_txt=preprocessed_instruction,
                output_txt=preprocessed_output
            )
            normalized_message = await parser_agent.run(parser_input)
            output_dict['message'] = normalized_message.output

        return result, output_dict

    except Exception as e:
        error_response = ShoppingResponse(
            message=f"-- ERROR: {str(e)}",
            base_random_keys=None,
            member_random_keys=None
        )
        return None, dict(error_response)
