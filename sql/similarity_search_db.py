# similarity_search.py

import os
import psycopg2
from openai import OpenAI
from dotenv import load_dotenv
from typing import Optional, List, Tuple
from psycopg2.extras import RealDictCursor
import base64
from io import BytesIO
from PIL import Image
import logging
import torch
from transformers import CLIPModel, CLIPProcessor

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

def load_clip_model(model_name="openai/clip-vit-base-patch32"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {DEVICE}")
    try:
        logging.info(f"Loading model: {model_name}...")
        model = CLIPModel.from_pretrained(model_name).to(DEVICE)
        processor = CLIPProcessor.from_pretrained(model_name)
        EMBEDDING_DIM = model.config.projection_dim
        logging.info(f"✅ Model initialized successfully. Embedding Dim: {EMBEDDING_DIM}")
        return model, processor, DEVICE, EMBEDDING_DIM
    except Exception as e:
        logging.critical(f"❌ Failed to initialize model: {e}", exc_info=True)

clip_model, clip_processor, DEVICE, EMBEDDING_DIM = load_clip_model()

def embed_base64_image(data_uri):
    """
    Convert a single base64 image string to a normalized CLIP embedding.
    Returns a 1D numpy array of shape (EMBEDDING_DIM,).
    """
    _, encoded = data_uri.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    with torch.no_grad():
        inputs = clip_processor(
            images=[image],
            return_tensors="pt",
            padding=True
        ).to(DEVICE)

        embedding = clip_model.get_image_features(**inputs)
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()[0]

def get_embedding(text):
    """Generate embedding vector for a given text using OpenAI."""
    response = client.embeddings.create(
        model=MODEL,
        input=text
    )
    return response.data[0].embedding

def similarity_search_image(data_uri, top_k: int = 5):
    query_vector = embed_base64_image(data_uri)
    query_vector_str = "[" + ",".join(map(str, query_vector)) + "]"

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT random_key,
                       persian_name,
                       1 - (embedding <=> %s) AS similarity
                FROM image_embedding
                ORDER BY embedding <=> %s
                LIMIT %s
            """, (query_vector_str, query_vector_str, top_k))

            results = cur.fetchall()

    return results

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

from typing import List, Optional, Any, Dict
import psycopg2

def find_candidate_shops(
    query: str,
    top_k: int = 1,
    price_min: Optional[int] = None,
    price_max: Optional[int] = None,
    **filters: Any,
) -> List[dict]:
    """
    Returns up to `top_k` candidate shops for a user query.
    - Uses product embeddings for similarity on Persian product name.
    - Respects filters with IS NULL OR ... conditions.
    - Special cases:
        * score → mt.score >= %(score)s
        * price_min/price_max → BETWEEN with ±5% tolerance
    """

    # Convert query to embedding
    query_vector = get_embedding(query)  # Returns list[float]
    query_vector_str = "[" + ",".join(map(str, query_vector)) + "]"

    # Price range defaults
    price_min_default, price_max_default = 0, 1000000000
    price_min = price_min if price_min is not None else price_min_default
    price_max = price_max if price_max is not None else price_max_default

    # ±5% tolerance
    price_min = int(price_min * 0.95) if price_min != price_max else price_min
    price_max = int(price_max * 1.05) if price_min != price_max else price_max

    params: Dict[str, Any] = {
        "query_vector": query_vector_str,
        "price_min": price_min,
        "price_max": price_max,
    }

    sql = """
    WITH filtered AS (
        SELECT 
            mt.base_random_key,
            mt.persian_name AS product_name,
            mt.shop_id,
            mt.price,
            mt.city,
            mt.has_warranty,
            mt.score,
            mt.extra_features,
            mt.member_random_key,
            mt.brand_title
        FROM member_total mt
        WHERE mt.price BETWEEN %(price_min)s AND %(price_max)s
    """
    # dynamic filters appended here…
    for key, value in filters.items():
        if value is None or value == "Ignore":
            continue
        if key=='has_warranty' and not value:
            value = None
        if key == "score":
            sql += f" AND (%({key})s IS NULL OR mt.score >= %({key})s)"
        else:
            sql += f" AND (%({key})s IS NULL OR mt.{key} = %({key})s)"
        params[key] = value

    sql += """
        ),
        ranked AS (
            SELECT 
                f.*,
                1 - (pe.embedding <=> %(query_vector)s::vector) AS similarity
            FROM filtered f
            JOIN product_embed pe ON f.base_random_key = pe.random_key
        )
        SELECT *
        FROM ranked
        ORDER BY similarity DESC
        LIMIT %(limit)s;
    """

    # sql += " LIMIT %(limit)s;"
    params["limit"] = top_k

    with psycopg2.connect(**DB_CONFIG) as conn:
        with conn.cursor() as cur:
            cur.execute("SET enable_seqscan = off;")
            cur.execute("SET ivfflat.probes = %s;", (20,))
            cur.execute(sql, params)
            rows = cur.fetchall()

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
            "brand_title": row[9],
            "similarity": round(row[10], 4) if row[10] is not None else None,
        }
        for row in rows
    ]
    print(results)
    return results
