# similarity_search.py

import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

OPENAI_API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = "text-embedding-3-small"

# --- Initialize OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY, base_url = BASE_URL)

def get_embedding(text):
    """Generate embedding vector for a given text using OpenAI."""
    response = client.embeddings.create(
        model=MODEL,
        input=text
    )
    return response.data[0].embedding
def similarity_search(query, top_k: int = 5, probes: int = 20):
    """
    Perform a similarity search in the product_embed table using pgvector IVFFlat index.

    Args:
        query (str): The query text.
        top_k (int): Number of similar items to return.
        probes (int): Number of IVF lists to probe (higher = better recall, slower).

    Returns:
        List of tuples: [(random_key, persian_name, similarity_score), ...]
    """
    query_vector = get_embedding(query)  # list[float]
    query_vector_str = "[" + ",".join(map(str, query_vector)) + "]"

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            # Force use of IVFFlat index
            cur.execute("SET enable_seqscan = off;")
            cur.execute("SET ivfflat.probes = %s;", (probes,))

            cur.execute("""
                SELECT random_key,
                       persian_name,
                       1 - (embedding <=> %s) AS similarity
                FROM product_embed
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_vector_str, query_vector_str, top_k))

            results = cur.fetchall()

    return results