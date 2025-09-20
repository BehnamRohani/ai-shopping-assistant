import os
from pathlib import Path
import psycopg2
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv


load_dotenv()

# -------------------------
# CONFIG
# -------------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}
data_path = "./data/"
QDRANT_PATH = data_path + "qdrant_db"
COLLECTION_NAME = "products"
EMBED_MODEL = "text-embedding-3-small"  # 1536 dims
VECTOR_DIM = 1536
BATCH_SIZE = 2000
CHECKPOINT_FILE = data_path + "qdrant_last_key.txt"
# -------------------------

# Load API credentials
OPENAI_API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

# Check if path exists and is non-empty
if Path(QDRANT_PATH).exists() and any(Path(QDRANT_PATH).iterdir()):
    print(f"Qdrant database already exists at {QDRANT_PATH}")
else:
    print(f"No Qdrant database found, will create new at {QDRANT_PATH}")

# Initialize persistent Qdrant client
client = QdrantClient(path=QDRANT_PATH)

# Create collection if needed
if not client.collection_exists(COLLECTION_NAME):
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"Created collection {COLLECTION_NAME} with dim={VECTOR_DIM}")
else:
    print(f"Collection {COLLECTION_NAME} already exists.")

# Wrap Qdrant client with LangChain VectorStore
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

# -------------------------
# Helper: checkpointing
# -------------------------
def load_checkpoint():
    p = Path(CHECKPOINT_FILE)
    return p.read_text(encoding="utf-8").strip() if p.exists() else None

def save_checkpoint(val):
    Path(CHECKPOINT_FILE).write_text(str(val), encoding="utf-8")


# -------------------------
# Main ingestion
# -------------------------
conn = psycopg2.connect(**DB_CONFIG)
cur = conn.cursor(name="product_cursor")
cur.itersize = BATCH_SIZE

last_key = load_checkpoint()
if last_key:
    print("Resuming from checkpoint:", last_key)
    cur.execute(
        "SELECT random_key, persian_name FROM base_products WHERE random_key > %s ORDER BY random_key ASC",
        (last_key,),
    )
else:
    cur.execute("SELECT random_key, persian_name FROM base_products ORDER BY random_key ASC")

total = 0
for batch in tqdm(iter(lambda: cur.fetchmany(BATCH_SIZE), []), desc="Batches", unit="batch"):
    if not batch:
        break

    texts = [row[1] for row in batch]
    metadatas = [{"random_key": row[0]} for row in batch]

    # Insert into Qdrant via LangChain
    vector_store.add_texts(texts=texts, metadatas=metadatas)

    # Save checkpoint
    save_checkpoint(batch[-1][0])
    total += len(batch)
    print(f"Inserted {total} rows (last random_key={batch[-1][0]})", end="\r")

print("\nDone.")
info = client.get_collection(COLLECTION_NAME)
print("Collection size:", info.vectors_count)