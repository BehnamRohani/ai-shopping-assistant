# similarity_search.py

import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List, Tuple
from psycopg2.extras import RealDictCursor

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

def find_candidate_shops(
    query: str,
    top_k: int = 3,
    has_warranty: Optional[bool] = None,
    score: Optional[int] = None,
    city_name: Optional[str] = None,
    brand_title: Optional[str] = None,
    price_min: int = None,
    price_max: int = None,
) -> List[dict]:
    """
    Returns up to `top_k` candidate shops for a user query.
    - Uses product embeddings for similarity on Persian product name.
    - Respects filters: warranty, score, city, brand, price.
    - None or 'Doesn''t Matter' = ignore filter.
    """
    # Convert query to embedding
    query_vector = get_embedding(query)  # Returns list[float]
    query_vector_str = "[" + ",".join(map(str, query_vector)) + "]"

    # Price range defaults
    price_min_default, price_max_default = 0, 100_000_000_000
    price_min = price_min if price_min is not None else price_min_default
    price_max = price_max if price_max is not None else price_max_default

    cur = conn.cursor(cursor_factory=RealDictCursor)

    sql = """
    WITH ranked_products AS (
        SELECT pe.random_key,
               1 - (pe.embedding <=> %s::vector) AS similarity
        FROM product_embed pe
    )
    SELECT 
        bp.persian_name AS product_name,
        m.shop_id,
        m.price,
        ci.name AS city,
        s.has_warranty,
        s.score,
        bp.extra_features,
        bp.random_key AS base_random_key,
        rp.similarity
    FROM ranked_products rp
    JOIN base_products bp ON rp.random_key = bp.random_key
    JOIN members m ON bp.random_key = m.base_random_key
    JOIN shops s ON m.shop_id = s.id
    JOIN brands b ON bp.brand_id = b.id
    JOIN cities ci ON s.city_id = ci.id
    WHERE 1=1
      AND (%(has_warranty)s IS NULL OR %(has_warranty)s = 'Doesn''t Matter' OR s.has_warranty = %(has_warranty)s)
      AND (%(score)s IS NULL OR %(score)s = 'Doesn''t Matter' OR s.score >= %(score)s)
      AND (%(city_name)s IS NULL OR %(city_name)s = 'Doesn''t Matter' OR ci.name = %(city_name)s)
      AND (%(brand_title)s IS NULL OR %(brand_title)s = 'Doesn''t Matter' OR b.title = %(brand_title)s)
      AND m.price BETWEEN %s * 0.95 AND %s * 1.05
    ORDER BY rp.similarity DESC, s.score DESC, m.price ASC
    LIMIT %s;
    """

    params = [
        query_vector_str,  # embedding
        has_warranty,
        score,
        city_name,
        brand_title,
        price_min,
        price_max,
        # query,  # product_name, can reuse the query
        # product_features,
        top_k
    ]
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            results = cur.fetchall()

    cur.close()
    conn.close()
    return results
