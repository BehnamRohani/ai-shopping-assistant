# fill_embeddings_batch_multi.py
import os
import psycopg2
from openai import OpenAI
from tqdm import tqdm
import psycopg2
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

OPENAI_API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
MODEL = "text-embedding-3-small"
BATCH_SIZE = 100  # number of rows per batch

# --- Initialize OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY, base_url = BASE_URL)

def get_rows_to_embed(cur, batch_size):
    cur.execute("""
        SELECT random_key, persian_name
        FROM product_embed
        WHERE embedding IS NULL
        LIMIT %s
    """, (batch_size,))
    return cur.fetchall()

with psycopg2.connect(**DB_CONFIG) as conn:
    with conn.cursor() as cur:
        # Get total number of rows to process for tqdm
        cur.execute("SELECT COUNT(*) FROM product_embed WHERE embedding IS NULL;")
        total_rows = cur.fetchone()[0]

        print(f"Total rows to update: {total_rows}")

        processed = 0
        with tqdm(total=total_rows, desc="Updating embeddings") as pbar:
            while True:
                rows = get_rows_to_embed(cur, BATCH_SIZE)
                if not rows:
                    break

                random_keys, texts = zip(*[(rk, txt) for rk, txt in rows if txt])

                # Generate embeddings for the batch
                response = client.embeddings.create(
                    model=MODEL,
                    input=list(texts)
                )

                embeddings = [item.embedding for item in response.data]

                # Prepare batch update
                update_data = list(zip(embeddings, random_keys))

                # Batch update in the DB
                cur.executemany("""
                    UPDATE product_embed
                    SET embedding = %s
                    WHERE random_key = %s
                """, update_data)
                conn.commit()

                processed += len(update_data)
                pbar.update(len(update_data))

print("All embeddings updated successfully.")
