import psycopg2
from psycopg2.extras import RealDictCursor
import os, re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": os.getenv("DB_PORT"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

def build_like_query(table_name: str, column_name: str, variable_query: str, limit: int = 10):
    """
    Build a parameterized SQL query for searching a table where a column
    contains a given substring (case-insensitive).

    Parameters
    ----------
    table_name : str
        Name of the table to query.
    column_name : str
        Name of the column to search in.
    variable_query : str
        Substring to search for.
    limit : int, optional
        Maximum number of rows to return (default is 10).

    Returns
    -------
    tuple[str, tuple]
        - query: The parameterized SQL query string.
        - params: Tuple of parameters to pass to execute() (pattern, limit).
    """
    pattern = f"%{variable_query}%"
    query = f"""
    SELECT *
    FROM {table_name}
    WHERE {column_name} ILIKE '{pattern}'
    LIMIT {limit};
    """
    return query


# -------------------------------
# Helper: execute SQL safely
# -------------------------------
def execute_sql(query: str):
    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                return cur.fetchall()
    except Exception as e:
        return f"-- ERROR executing query: {str(e)}"
    


def extract_sql(text: str) -> str:
    """
    Extracts the SQL query from an LLM/agent response.
    Handles fenced code blocks (```sql ... ```), plain SQL, and comments.
    """
    if not text:
        return ""

    # 1. If inside ```sql ... ```
    code_block = re.search(r"```(?:sql)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if code_block:
        return code_block.group(1).strip().rstrip(";") + ";"

    # 2. If inside generic ``` ```
    generic_block = re.search(r"```([\s\S]*?)```", text)
    if generic_block:
        return generic_block.group(1).strip().rstrip(";") + ";"

    # 3. Otherwise, assume text is already SQL
    return text.strip().rstrip(";") + ";"