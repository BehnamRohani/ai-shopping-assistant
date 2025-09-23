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
import pickle
from dotenv import load_dotenv
import httpx
from pydantic_ai import Agent, UsageLimits, ModelSettings, BinaryContent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from sql.sql_agent import generate_sql_query
from sql.sql_utils import execute_sql
from prompt.prompts import *
from utils.utils import preprocess_persian
from sql.similarity_search_db import similarity_search
from sql.sql_utils import get_chat_history, get_base_id_and_index
from typing import Tuple, Optional


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
IMAGE_MODEL = os.getenv("IMAGE_MODEL")

OPENAI_API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")  # e.g. https://turbo.torob.com/v1

initial_client = OpenAIChatModel(
    INITIAL_THOUGHT_MODEL,
    provider=OpenAIProvider(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        http_client=httpx.AsyncClient()
    ),
    settings=ModelSettings(temperature=0.0001, max_tokens=1024)
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
    - finished: Boolean indicating if the answer is definite / final.
    """
    message: Optional[str] = None
    base_random_keys: Optional[List[str]] = None
    member_random_keys: Optional[List[str]] = None
    finished: bool = False


shopping_client = OpenAIChatModel(
    SHOPPING_MODEL,
    provider=OpenAIProvider(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        http_client=httpx.AsyncClient()
    ),
    settings=ModelSettings(temperature=0.0001, max_tokens=1024)
)

shopping_agent = Agent(
    name="TorobShoppingAssistant",
    model=shopping_client,
    system_prompt=system_prompt.format(system_role = system_role,
                                        tools = tools_info, 
                                       rules = instructions_generated),
    tools=[similarity_search,
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

async def run_shopping_agent(
    input_dict: dict,
    use_initial_plan: bool = True,
    use_parser_output: bool = True,
    use_initial_similarity_search: bool = True,
) -> Tuple[Optional[ShoppingResponse], dict]:
    """
    Execute the Torob Shopping Assistant pipeline for a given user instruction/chat input.

    Args:
        input_dict (dict): User input as a dictionary, e.g.,
            {
              "chat_id": "a1b2c3d4-e5f6-7890-1234-567890abcdef",
              "messages": [
                {
                  "type": "text",
                  "content": "سلام، یک گوشی آیفون ۱۶ پرومکس میخوام"
                }
              ]
            }
        use_initial_plan (bool): If True, generate a high-level plan via CoT agent.
        use_parser_output (bool): If True, normalize final message using parser agent.
        use_initial_similarity_search (bool): If True, run similarity search first and include results in prompt.

    Steps:
    1. Preprocess the user input (Persian normalization).
    2. Optionally generate a high-level plan using the CoT agent.
    3. Optionally run similarity search and append candidate products.
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
        # Step 0: Get text and image Message
        all_texts = [m["content"] for m in input_dict["messages"] if m["type"] == "text"]
        all_images = [m["content"] for m in input_dict["messages"] if m["type"] == "image"]
        instruction = all_texts[0] if all_texts else None
        user_image = all_images[0] if all_images else None

        # If image request, then handle using image client
        if user_image:
            result, image_response = await run_image_agent(input_text = instruction,
                                  image_b64 = user_image,
                                  usage_limits = usage_limits)
            print(image_response)
            final_dict = ShoppingResponse(
                            message=image_response['main_topic'],
                            base_random_keys=None,
                            member_random_keys=None,
                            finished=True,
                            )
            return result, dict(final_dict)

        # Step 0.5: fetch chat history
        chat_id = input_dict["chat_id"]
        base_id, chat_index = get_base_id_and_index(chat_id)   # your existing function
        history = get_chat_history(base_id)[-4:]

        # Convert to readable string for the LLM
        history_text = ""
        if history:
            history_text = "\n".join(
                [f"Input No.({(i+1)}): {h['message']}\nResponse No.({(i+1)}): {h['response']}" for i,h in enumerate(history)]
            )
        print(history_text)

        # Step 1: preprocess input
        preprocessed_instruction = preprocess_persian(instruction)

        # Step 2: optionally run similarity search
        similarity_text = ""
        if use_initial_similarity_search and (history_text==""):
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

        # Step 3: build prompt for shopping agent
        prompt = ""
        # add history if exists
        if history_text:
            prompt += "Conversation history:\n" + history_text + "\n\n"
        # If this is the fifth turn (end), alert the model
        if chat_index ==5:
            prompt += "[IMPORTANT] This is the Fifth turn. Your response is the end of conversation. You must answer the user now definitively.\n"
        prompt += "Input: " + preprocessed_instruction

        # Add initial similarity optionally
        if similarity_text:
            prompt += "\n\nInitial Similarity Search Candidates:\n" + similarity_text
            prompt += "\n" + "The initial similarity search results are provided for convenience."

        # Step 4: optionally generate plan
        plan_output = None
        plan_text = ""
        if use_initial_plan and (chat_index == 1 or chat_index == 5): # for the 1st and last turn
            plan_output = await cot_agent.run(prompt)
            plan_text = plan_output.output or ""
            print(plan_text)
        # Add plan
        if plan_text:
            prompt += "\n\nPlan:\n" + plan_text
        
        # Run main agent
        result = await shopping_agent.run(prompt, usage_limits=usage_limits)

        # Convert to dict for output formatting
        output_dict = dict(result.output)

        # Step 5: optionally normalize message
        message_out = result.output.message
        if message_out and use_parser_output and result.output.finished and (not history_text): # only if the output is final
            preprocessed_output = preprocess_persian(message_out)
            parser_input = parser_prompt.format(
                input_txt=preprocessed_instruction,
                output_txt=preprocessed_output
            )
            normalized_message = await parser_agent.run(parser_input)
            output_dict['message'] = normalized_message.output
        
        # Handling
        output_dict['base_random_keys'] = None if not result.output.finished else output_dict['base_random_keys']
        output_dict['member_random_keys'] = None if not result.output.finished else output_dict['member_random_keys']

        return result, output_dict

    except Exception as e:
        error_response = ShoppingResponse(
            message=f"-- ERROR: {str(e)}",
            base_random_keys=None,
            member_random_keys=None,
            finished=True,
        )
        return None, dict(error_response)

import base64
from typing import Tuple

def extract_media_type_and_bytes(data_uri: str) -> Tuple[str, bytes]:
    """
    Extracts the media type and image bytes from a base64 data URI.

    Args:
        data_uri: string like "data:image/jpeg;base64,<image-base64>"

    Returns:
        Tuple of (media_type, image_bytes)
    """
    if not data_uri.startswith("data:"):
        raise ValueError("Invalid data URI format")

    # Split header and base64 part
    header, b64_data = data_uri.split(",", 1)

    # Extract media type
    # Example header: "data:image/jpeg;base64"
    media_type = header.split(";")[0][5:]  # remove "data:"

    # Decode base64 to bytes
    image_bytes = base64.b64decode(b64_data)

    return image_bytes, media_type

image_client = OpenAIChatModel(
    IMAGE_MODEL,
    provider=OpenAIProvider(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        http_client=httpx.AsyncClient()
    ),
    settings=ModelSettings(temperature=0.0001, max_tokens=1024)
)


# --- Output Schema ---
class ImageResponse(BaseModel):
    description: Optional[str] = None
    long_description: Optional[str] = None
    candidates: Optional[List[str]] = None
    main_topic: Optional[str] = None

# --- Agent ---

import pickle
with open("categories_by_level.pkl", "rb") as f:
    loaded_levels = pickle.load(f)

# for lvl, cats in loaded_levels.items():
#     print(f"Level {lvl}:", cats[:10], "...")  # show first 10 per level
labels_quotes = [f"Level {lvl}: {cats}" "\n" for lvl,cats in loaded_levels.items()]


image_agent = Agent(
    name="TorobImageAssistant",
    model=image_client,
    system_prompt=image_label_system_prompt + "\n" + "\n".join(labels_quotes),
    output_type=ImageResponse,
)


# --- Runner function ---
async def run_image_agent(
    input_text: str,
    image_b64: str,
    usage_limits: Optional[Any] = None,
) -> Tuple[Optional[ImageResponse], Dict[str, Any]]:
    """
    Run the image agent with message content (text + image_b64).
    """
    try:
        # Compose message payload exactly like OpenAI API
        image_bytes, media_type = extract_media_type_and_bytes(image_b64)
        
        user_message = [
                input_text, 
                BinaryContent(data=image_bytes, media_type=media_type),
                        ]

        if usage_limits:
            result = await image_agent.run(user_message, usage_limits=usage_limits)
        else:
            result = await image_agent.run(user_message)

        output_obj = result.output
        return result, dict(output_obj)

    except Exception as e:
        error = {
            "description": None,
            "long_description": None,
            "candidates": [],
            "main_topic": None,
            "message": f"-- ERROR: {str(e)}"
        }
        return None, error
