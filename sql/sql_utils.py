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

def build_exact_query_and_execute(
    table_name: str,
    column_name: str,
    variable_query: str,
    limit: int = 10,
    columns: list[str] | None = None,
):
    """
    Build and execute a SQL query for searching a table where a column
    exactly matches a given value.

    Parameters
    ----------
    table_name : str
        Name of the table to query.
    column_name : str
        Name of the column to filter on.
    variable_query : str
        Exact value to match.
    limit : int, optional
        Maximum number of rows to return (default is 10). Must be >= 1.
    columns : list[str], optional
        List of column names to return. Defaults to ["*"].

    Returns
    -------
    list[RealDictRow] | str
        Query results if successful, or error string if execution fails.
    """
    if limit < 3:
        limit = 1  # enforce minimum

    if not columns:
        columns = ["*"]

    cols_str = ", ".join(columns)

    query = f"""
    SELECT {cols_str}
    FROM {table_name}
    WHERE {column_name} = %s
    LIMIT %s;
    """
    params = (variable_query, limit)

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchall()
    except Exception as e:
        return f"-- ERROR executing query: {str(e)}"


def build_like_query_and_execute(
    table_name: str,
    column_name: str,
    variable_query: str,
    limit: int = 10,
    columns: list[str] | None = None,
):
    """
    Build and execute a SQL query for searching a table where a column
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
        Maximum number of rows to return (default is 10). Must be >= 3.
    columns : list[str], optional
        List of column names to return. Defaults to ["*"].

    Returns
    -------
    list[RealDictRow] | str
        Query results if successful, or error string if execution fails.
    """
    if limit < 3:
        limit = 3  # enforce minimum

    if not columns:
        columns = ["*"]

    cols_str = ", ".join(columns)
    pattern = f"%{variable_query}%"

    query = f"""
    SELECT {cols_str}
    FROM {table_name}
    WHERE {column_name} ILIKE %s
    LIMIT %s;
    """
    params = (pattern, limit)

    try:
        with psycopg2.connect(**DB_CONFIG) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params)
                return cur.fetchall()
    except Exception as e:
        return f"-- ERROR executing query: {str(e)}"


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