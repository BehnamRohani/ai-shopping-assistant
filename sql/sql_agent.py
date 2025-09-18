from pydantic_ai import Agent, ModelSettings
import httpx
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import os
from sql.sql_utils import extract_sql
from prompt.prompts import system_prompt_sql

OPENAI_API_KEY = os.environ['API_KEY']
BASE_URL = os.environ['BASE_URL']
MODEL_NAME = "gpt-4.1-mini"

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
    settings=ModelSettings(temperature=0.001, max_tokens=1024)
)

# -------------------------------
# Define the Agent
# -------------------------------
sql_agent = Agent(
    name="TorobSQLAgent",
    model=client,
    system_prompt=system_prompt_sql,
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
