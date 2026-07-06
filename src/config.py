# src/config.py
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")