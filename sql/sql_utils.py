import psycopg2
from psycopg2.extras import RealDictCursor
import os, re
import random
import string
from datetime import datetime
import json
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

def generate_base_id(length: int = 12) -> str:
    """Generate a random 12-character base_id."""
    return "".join(random.choices(string.ascii_letters + string.digits, k=length))

def get_base_id_and_index(chat_id: str) -> tuple[str, int]:
    """
    Determine the base_id and chat_index for a given chat_id.
    
    Logic:
    - Fetch the latest row for this chat_id ordered by timestamp descending.
    - If no row exists or latest row is finished → generate new base_id, set chat_index=1.
    - Else → reuse base_id and increment chat_index by 1.
    """
    conn = get_db_conn()
    cur = conn.cursor()

    # Fetch the most recent message for this chat_id
    cur.execute("""
        SELECT base_id, chat_index, finished
        FROM chats
        WHERE chat_id = %s
        ORDER BY timestamp DESC
        LIMIT 1
    """, (chat_id,))
    row = cur.fetchone()

    if row is None or row[2]:  # No previous rows OR finished=True
        base_id = generate_base_id()
        chat_index = 1
    else:  # unfinished session exists
        base_id = row[0]
        chat_index = row[1] + 1

    conn.commit()
    cur.close()
    conn.close()

    return base_id, chat_index


# ------ DB Insert Helper ------
def insert_chat(input_dict: dict, output_dict: dict):
    """
    Insert a chat message into the 'chats' table.
    Each row represents the last user message + model response at this index.
    """
    chat_id = input_dict["chat_id"]   # treat chat_id as base_id

    # --- Extract user message fully ---
    all_texts = [m["content"] for m in input_dict["messages"] if m["type"] == "text"]
    all_images = [m["content"] for m in input_dict["messages"] if m["type"] == "image"]

    user_image_url = all_images[0] if all_images else None
    user_text = all_texts[0] if all_texts else None

    finished = bool(output_dict.get("member_random_keys"))

    base_id, chat_index = get_base_id_and_index(chat_id=chat_id)

    # --- Prepare row ---
    row = {
        "chat_id": chat_id,
        "base_id": base_id,
        "user_text": user_text,
        "user_image_url": user_image_url,
        "chat_index": chat_index,
        "model_text": output_dict.get("message"),
        "model_image_url": None,
        "base_random_keys": json.dumps(output_dict.get("base_random_keys")),
        "member_random_keys": json.dumps(output_dict.get("member_random_keys")),
        "finished": finished,
    }

    sql = """
    INSERT INTO chats (
        chat_id, base_id, user_text, user_image_url, chat_index,
        model_text, model_image_url, base_random_keys, member_random_keys, finished
    )
    VALUES (%(chat_id)s, %(base_id)s, %(user_text)s, %(user_image_url)s, %(chat_index)s,
            %(model_text)s, %(model_image_url)s, %(base_random_keys)s, %(member_random_keys)s, %(finished)s)
    """
    conn = get_db_conn()
    cur = conn.cursor()
    cur.execute(sql, row)   # <-- replace with your DB exec layer
    conn.commit()
    cur.close()
    conn.close()


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