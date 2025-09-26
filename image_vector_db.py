# fill_embeddings_from_file.py
import os
import psycopg2
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

JSONL_FILE = "data/image_embedding.jsonl"
BATCH_SIZE = 100  # rows per DB update


with psycopg2.connect(**DB_CONFIG) as conn:
    with conn.cursor() as cur:
        # Enable pgvector extension if not already installed
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # Create the image_embedding table if it doesn't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS image_embedding (
                random_key TEXT PRIMARY KEY,
                embedding vector(768)
            );
        """)
        conn.commit()

def load_embeddings_from_jsonl(file_path):
    """Load embeddings from a .jsonl file into a dict {random_key: embedding}"""
    data = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("random_key")
            embedding = record.get("embedding")
            if key and embedding:
                data[key] = embedding
    return data

# --- Main ---
embeddings_map = load_embeddings_from_jsonl(JSONL_FILE)
print(f"Loaded {len(embeddings_map)} embeddings from file.")

with psycopg2.connect(**DB_CONFIG) as conn:
    with conn.cursor() as cur:
        keys = list(embeddings_map.keys())
        with tqdm(total=len(keys), desc="Updating embeddings") as pbar:
            for i in range(0, len(keys), BATCH_SIZE):
                batch_keys = keys[i:i+BATCH_SIZE]
                update_data = [(k, embeddings_map[k]) for k in batch_keys]

                # Update database (embedding must be cast to vector(768))
                cur.executemany("""
                    INSERT INTO image_embedding (random_key, embedding)
                    VALUES (%s, %s::vector(768))
                    ON CONFLICT (random_key) DO NOTHING;
                """, update_data)
                conn.commit()
                pbar.update(len(batch_keys))

print("All embeddings updated successfully from JSONL.")
