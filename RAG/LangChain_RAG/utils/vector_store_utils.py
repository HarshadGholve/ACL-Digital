from chromadb import Client
from chromadb.config import Settings

# Initialize Chroma client with duckdb+parquet implementation
client = Client(
    Settings(
        chroma_db_impl="duckdb+parquet",    # Using duckdb+parquet for local storage
        persist_directory="./data/vector_store"  # Directory for persisting data
    )
)

# Create or load the collection
collection = client.get_or_create_collection(name="document_store")

# Define the variable all_splits
all_splits = [
    {"id": "1", "content": "Document 1 content"},
    {"id": "2", "content": "Document 2 content"}
]

# Add documents to the collection
collection.add(documents=all_splits)

# Persist the data to disk
client.persist()

print("Data has been persisted to disk.")
