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

def create_member_total_view():
    """
    Create the member_total view combining base_products, members, shops, brands, and cities.
    """
    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP VIEW IF EXISTS member_total CASCADE;")
            cur.execute("""
                CREATE OR REPLACE VIEW member_total AS
                SELECT 
                    bp.random_key AS base_random_key,
                    bp.persian_name,
                    bp.extra_features,
                    m.shop_id,
                    m.price,
                    m.random_key AS member_random_key,
                    s.score,
                    s.has_warranty,
                    b.title AS brand_title,
                    ci.name AS city
                FROM base_products bp
                JOIN members m ON bp.random_key = m.base_random_key
                JOIN shops s ON m.shop_id = s.id
                JOIN brands b ON bp.brand_id = b.id
                JOIN cities ci ON s.city_id = ci.id;
                """)
        conn.commit()

OPENAI_API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = "text-embedding-3-small"

# --- Initialize OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY, base_url = BASE_URL)

create_member_total_view()

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

def similarity_search_cat(query, top_k: int = 5):
    """
    Perform a similarity search in the categories table using pgvector.

    Args:
        query (str): The query text.
        top_k (int): Number of similar categories to return.

    Returns:
        List[Dict]: [
            {"category": str, "similarity": float}, ...
        ]
    """
    query_vector = get_embedding(query)  # list[float]
    query_vector_str = "[" + ",".join(map(str, query_vector)) + "]"

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            # Force use of IVFFlat index
            cur.execute("""
                SELECT c.title,
                       1 - (c.embedding <=> %s) AS similarity
                FROM categories c
                ORDER BY c.embedding <=> %s
                LIMIT %s
            """, (query_vector_str, query_vector_str, top_k))

            results = cur.fetchall()

    results = [
        {"category": row[0], "similarity": round(row[1], 4)}
        for row in results
    ]
    return results


def find_candidate_shops(
    query: str,
    top_k: int = 1,
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
    price_min_default, price_max_default = 0, 1000000000
    price_min = price_min if price_min is not None else price_min_default
    price_max = price_max if price_max is not None else price_max_default

    price_min = int(price_min * 0.95)
    price_max = int(price_max * 1.05)

    sql = """
        WITH ranked_products AS (
                SELECT random_key, 1 - (embedding <=> %(query_vector)s::vector) AS similarity
                FROM product_embed
                ORDER BY embedding <=> %(query_vector)s::vector
            )
        SELECT 
            mt.persian_name AS product_name,
            mt.shop_id,
            mt.price,
            mt.city,
            mt.has_warranty,
            mt.score,
            mt.extra_features,
            mt.base_random_key,
            mt.member_random_key,
            rp.similarity
        FROM ranked_products rp
        JOIN member_total mt ON rp.random_key = mt.base_random_key
        WHERE 1=1
            AND (%(has_warranty)s IS NULL OR mt.has_warranty = %(has_warranty)s)
            AND (%(score)s IS NULL OR mt.score >= %(score)s)
            AND (%(city_name)s IS NULL OR mt.city = %(city_name)s)
            AND (%(brand_title)s IS NULL OR mt.brand_title = %(brand_title)s)
            AND mt.price BETWEEN %(price_min)s AND %(price_max)s
        ORDER BY rp.similarity DESC, mt.score DESC, mt.price ASC
        LIMIT %(limit)s;
        """

    params = {
        "query_vector": query_vector_str,
        "has_warranty": has_warranty,
        "score": score,
        "city_name": city_name,
        "brand_title": brand_title,
        "price_min": price_min,
        "price_max": price_max,
        "limit": top_k,
    }

    for k,v in params.items():
        if v in ['Ignore']:
            params[k] = None

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off;")
            cur.execute("SET ivfflat.probes = %s;", (20,))

            cur.execute(sql, params)
            results = cur.fetchall()
    results = [
        {
            "product_name": row[0], 
            "shop_id": row[1],
            "price": row[2],
            "city_name": row[3],
            "has_warranty": row[4],
            "score": row[5],
            "extra_features": row[6],
            "base_random_key": row[7],
            "member_random_key": row[8],
            "similarity": round(row[9], 4),
         }
        for row in results
    ]
    print(results)
    return results
