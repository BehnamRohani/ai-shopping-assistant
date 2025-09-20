import os
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from openai import OpenAI
from dotenv import load_dotenv

import os
import psycopg2
from psycopg2.extras import RealDictCursor


load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}
# -------------------------
# CONFIG
# -------------------------
QDRANT_PATH = "./notebooks/qdrant_db"  # local disk folder
COLLECTION_NAME = "products"
EMBED_MODEL = "text-embedding-3-small"
OPENAI_API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
TOP_K = 5

# Load API credentials
OPENAI_API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

# Initialize embeddings
embeddings = OpenAIEmbeddings(
    model=EMBED_MODEL,
    api_key=OPENAI_API_KEY,
    base_url=BASE_URL,
)

# Connect to local disk-based Qdrant
client = QdrantClient(path=QDRANT_PATH)

# Wrap with LangChain VectorStore
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)

def get_persian_name_from_db(random_key):
    """
    Fetch the Persian name from the database using the random_key.
    
    Returns the name as a string if found, otherwise returns None.
    """
    if not random_key:
        return None

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        # Replace 'your_table' and 'persian_name_column' with actual table/column names
        query = """
            SELECT persian_name
            FROM base_products
            WHERE random_key = %s
            LIMIT 1;
        """
        cursor.execute(query, (random_key,))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        if result and "persian_name" in result:
            return result["persian_name"]
        return None

    except Exception as e:
        print(f"Error querying Persian name for random_key {random_key}: {e}")
        return None


while True:
    try:
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() in ("exit", "quit"):
            print("Exiting.")
            break
        if not query:
            print("Empty query. Try again.")
            continue

        results = vector_store.similarity_search(query, k=TOP_K)

        if not results:
            print("No results found.")
            continue

        print(f"\nTop {TOP_K} similar names:")
        for i, doc in enumerate(results, start=1):
            # Try to get Persian name, fallback to deriving from DB
            name = doc.metadata.get("persian_name")
            if not name:
                try:
                    # Replace this with your actual DB lookup function
                    name = get_persian_name_from_db(doc.metadata.get("random_key"))
                except Exception as e:
                    name = "N/A"
                    print(f"Error deriving Persian name for random_key {doc.metadata.get('random_key')}: {e}")

            rk = doc.metadata.get("random_key", "N/A")
            print(f"{i}. {name}  (random_key: {rk})")

    except Exception as e:
        print(f"An error occurred during the query: {e}")
