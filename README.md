# AI Shopping Assistant

## Project Overview

This repository contains an AI Shopping Assistant designed to analyze user instructions and retrieve product information, leveraging data from Torob.com.  The assistant utilizes Python, along with specific frameworks and tools, to provide detailed product recommendations and answer user queries.

## Key Features & Benefits

- **Intelligent Product Search:** Analyzes user queries to identify relevant products.
- **Detailed Product Information:** Retrieves and presents comprehensive product details from Torob.com.
- **AI-Powered Recommendations:** Uses AI to provide personalized product suggestions.
- **Dockerized Deployment:** Easy deployment using Docker.
- **Integration with Torob.com:** Leverages data from Torob.com's product database.

## Prerequisites & Dependencies

Before you begin, ensure you have the following installed:

- **Python 3.13:**  The project is built using Python 3.13.
- **Docker:** Docker is required for containerization and deployment.
- **pip:**  Python package installer for managing dependencies.
- **PostgreSQL:** A PostgreSQL database is required to store product data and embeddings.

Python dependencies are managed using `requirements.txt` and can be installed using pip. The project uses the following libraries:

- `httpx`
- `pydantic`
- `pydantic_ai`
- `python-dotenv`
- `fastapi`
- `psycopg2`
- `torch`
- `torchvision`
- `torchaudio`

## Installation & Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/BehnamRohani/ai-shopping-assistant.git
   cd ai-shopping-assistant
   ```

2. **Set up Environment Variables:**

   - Create a `.env` file in the root directory.
   - Add the following environment variables with your specific values:

     ```
     DB_HOST=<your_db_host>
     DB_PORT=<your_db_port>
     DB_NAME=<your_db_name>
     DB_USER=<your_db_user>
     DB_PASSWORD=<your_db_password>
     OPENAI_API_KEY=<your_openai_api_key>
     TOROB_TOKEN=<your_torob_token> # Optional, if needed
     ```

3. **Install Python Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Build and Run the Docker Container:**

   ```bash
   docker build -t ai-shopping-assistant .
   docker run -p 8000:8000 ai-shopping-assistant
   ```
   This will build a docker image named `ai-shopping-assistant` and run the application, exposing it on port 8000.  You can then access the application in your browser at `http://localhost:8000`.

5. **Database Setup:**

   - Ensure that your PostgreSQL database is running and accessible.
   - Run the `image_vector_db.py` script to initialize the database with the necessary tables and data (including image embeddings, replace `<your_embedding_file>` with path):

     ```bash
     python image_vector_db.py
     ```
   - It's crucial to populate the database with product data and image embeddings for the assistant to function correctly.  Refer to `sql/similarity_search_db.py` and `sql/sql_utils.py` for database schema details.

## Usage Examples & API Documentation

The application exposes a FastAPI endpoint that can be used to interact with the AI Shopping Assistant.

### API Endpoints

The `app.py` file contains the API endpoints. Example of how the API can be used (adjust based on actual API definition):

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

class UserQuery(BaseModel):
    query: str

@app.post("/search")
async def search(user_query: UserQuery):
    # Your logic to process the query using the AI Shopping Assistant
    # and return the search results.
    try:
        # Example:
        results = search_products(user_query.query)  # Replace with your function
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

```

To use this example (after running the docker container), send a POST request to `http://localhost:8000/search` with the following JSON body:

```json
{
  "query": "best gaming laptop under $1500"
}
```

The API will return a JSON response containing the search results.

## Configuration Options

The AI Shopping Assistant can be configured using environment variables defined in the `.env` file.  Key configuration options include:

- **Database Credentials:**  `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
- **API Keys:** `OPENAI_API_KEY`, `TOROB_TOKEN` (if applicable)

You can also adjust the behavior of the AI agents by modifying the prompts defined in the `prompt/prompts.py` file.

## Contributing Guidelines

We welcome contributions to the AI Shopping Assistant project! To contribute:

1. **Fork the repository.**
2. **Create a new branch** for your feature or bug fix.
3. **Make your changes** and ensure they are well-documented.
4. **Submit a pull request** with a clear description of your changes.

Please follow the existing code style and conventions.  Ensure that your code is well-tested and includes appropriate unit tests.

## License Information

This project has no license specified. All rights are reserved to the owner, BehnamRohani.

## Acknowledgments

- Thanks to the Torob.com team for providing the product data.
- This project leverages the `pydantic-ai` library for AI model integration.
