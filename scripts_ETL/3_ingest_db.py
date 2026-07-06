

""" Chunks the data and pushes it to your Weaviate Vector Database """


import os
import pandas as pd
import weaviate
from weaviate.auth import AuthApiKey
from weaviate.classes.config import Configure, Property, DataType
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import sys

# Adding the root directory to the system path so we can import from src.config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import WEAVIATE_URL, WEAVIATE_API_KEY, HF_TOKEN

def load_portfolio_docs(file_path="github_portfolio_dataset.csv"):
    df = pd.read_csv(file_path)
    documents = []

    for _, row in df.iterrows():
        page_content = (
            f"Project Name: {row['project']}\n"
            f"Project Type: {row['project_type']}\n"
            f"Summary: {row['summary']}\n"
            f"Skills/Tech Stack: {row['skills']}\n"
            f"Key Features: {row['key_features']}\n"
            f"Concepts: {row['concepts']}"
        )
        metadata = {
            "project_name": str(row["project"]),
            "github_url": str(row["github_url"]),
            "project_type": str(row["project_type"])
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
    
    print(f"Loaded {len(documents)} portfolio items from CSV.")
    return documents

def main():
    print("Connecting to Weaviate Cloud...")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=AuthApiKey(WEAVIATE_API_KEY),
        headers={"X-HuggingFace-Api-Key": HF_TOKEN}
    )

    if not client.is_ready():
        print("Weaviate is not ready!")
        return

    # 1. Create Collection
    collection_name = "Portfolio"
    
    # If it already exists, you can delete it to start fresh, or skip. 
    # Here we delete it to ensure a fresh upload.
    if client.collections.exists(collection_name):
        print(f"Collection '{collection_name}' already exists. Deleting to rebuild...")
        client.collections.delete(collection_name)

    print(f"Creating '{collection_name}' collection...")
    client.collections.create(
        name=collection_name,
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="project_name", data_type=DataType.TEXT),
            Property(name="project_type", data_type=DataType.TEXT),
            Property(name="github_url", data_type=DataType.TEXT)
        ],
        vectorizer_config=Configure.Vectorizer.none() # We compute embeddings manually below
    )

    # 2. Load Docs & Embedding Model
    portfolio_docs = load_portfolio_docs("data/github_portfolio_dataset.csv") # Update path if needed
    
    print("Loading HuggingFace Embeddings Model (this might take a moment)...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 3. Ingest Data into Weaviate
    collection = client.collections.get(collection_name)
    
    print("Generating embeddings and uploading to Weaviate...")
    with collection.batch.dynamic() as batch:
        for doc in portfolio_docs:
            # Generate the embedding vector
            vector = embedding_model.embed_query(doc.page_content)
            
            # Upload the data + vector
            batch.add_object(
                properties={
                    "content": doc.page_content,
                    "project_name": doc.metadata["project_name"],
                    "project_type": doc.metadata["project_type"],
                    "github_url": doc.metadata["github_url"]
                },
                vector=vector
            )
            
    if len(collection.batch.failed_objects) > 0:
        print("Some objects failed to insert!")
    else:
        print("Database successfully populated with the portfolio!")

    client.close()

if __name__ == "__main__":
    main()
