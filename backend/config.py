"""
Configuration for the Document Chatbot system.
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Embedding Model
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Chunking
CHUNK_SIZE = 800          # characters per chunk (optimized for Vietnamese)
CHUNK_OVERLAP = 150       # overlap between chunks

# Search
TOP_K = 5                 # number of results to return
SIMILARITY_THRESHOLD = 40 # minimum similarity % to include in results

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".txt"}

# Server
HOST = "0.0.0.0"
PORT = 8000
