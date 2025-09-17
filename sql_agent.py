from pydantic_ai import Agent
import httpx
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import os
from sql_utils import extract_sql

OPENAI_API_KEY = os.environ['API_KEY']
BASE_URL = os.environ['BASE_URL']
MODEL_NAME = "gpt-4.1-mini"

# -------------------------------
# SYSTEM PROMPT
# -------------------------------
system_prompt = """
You are an expert SQL assistant for torob.com. 
Your job is to generate **PostgreSQL SQL queries** from user instructions. 

### Guidelines:
- Always return valid PostgreSQL queries.
- Use only the tables and columns provided in the schema.
- Do not invent columns or tables that do not exist.
- Use clear aliases where useful.
- Join tables only when logically necessary.
- Prefer descriptive SELECT clauses over SELECT *.
- Always think step by step about the schema before writing the query.

### Database Schema:
Tables and their columns:

1. searches(id, uid, query, page, timestamp, session_id, result_base_product_rks, category_id, category_brand_boosts)
2. base_views(id, search_id, base_product_rk, timestamp)
3. final_clicks(id, base_view_id, shop_id, timestamp)
4. base_products(random_key, persian_name, english_name, category_id, brand_id, extra_features, image_url, members)
5. members(random_key, base_random_key, shop_id, price)
6. shops(id, city_id, score, has_warranty)
7. categories(id, title, parent_id)
8. brands(id, title)
9. cities(id, name)

### Output Format:
ALWAYS wrap the sql code in ```sql```.
"""

# -------------------------------
# Define the model with custom base URL, API key, and model name
# -------------------------------
client = OpenAIChatModel(
    MODEL_NAME,
    provider=OpenAIProvider(
        base_url=BASE_URL,
        api_key=OPENAI_API_KEY,
        http_client=httpx.AsyncClient()
    ),
)

# -------------------------------
# Define the Agent
# -------------------------------
sql_agent = Agent(
    name="TorobSQLAgent",
    model=client,
    system_prompt=system_prompt,
)


async def generate_sql_query(instruction: str) -> str:
    """
    Generate a PostgreSQL SQL query from a natural language instruction using the SQL agent.

    This function:
      1. Sends the instruction to the `sql_agent`.
      2. Extracts a clean SQL query string from the agent's output 
         (removing ```sql fences, trailing semicolons, etc.).
      3. Returns both the cleaned SQL query and the raw agent output.

    Parameters
    ----------
    instruction : str
        A natural language description of the desired query (e.g., 
        "Show the 5 most recent searches with their queries and timestamps.").

    Returns
    -------
    tuple[str, str]
        (sql_query, raw_output) if successful:
            - sql_query: the cleaned, executable SQL query.
            - raw_output: the unprocessed response from the agent.
        
        If an exception occurs, returns an error message string prefixed with `-- ERROR`.

    Notes
    -----
    - This function uses the helper `extract_sql()` to sanitize the SQL query.
    - Ensure `sql_agent` and `extract_sql` are defined and imported.
    """
    try:
        result = await sql_agent.run(instruction)
        sql_query = extract_sql(result.output)
        return sql_query, result.output
    except Exception as e:
        # Handle model/agent errors gracefully
        return "", f"-- ERROR: Failed to generate SQL query.\n-- Reason: {str(e)}"
