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

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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

def similarity_search(query, top_k=5):
    """
    Perform a similarity search in the product_embed table.
    
    Args:
        query (str): The query text.
        top_k (int): Number of similar items to return.
    
    Returns:
        List of tuples: [(random_key, persian_name, similarity_score), ...]
    """
    query_vector = get_embedding(query)

    results = []
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            # Use pgvector cosine similarity
            cur.execute(f"""
                SELECT random_key, persian_name, 1 - (embedding <=> %s) AS similarity
                FROM product_embed
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_vector, query_vector, top_k))
            
            results = cur.fetchall()

    return results