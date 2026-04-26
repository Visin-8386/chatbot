"""
Configuration for the Document Chatbot system.
"""
import os
from typing import List

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)


def _parse_csv_env(name: str, default: List[str]) -> List[str]:
	raw_value = os.getenv(name, "").strip()
	if not raw_value:
		return default

	values = [item.strip() for item in raw_value.split(",") if item.strip()]
	return values or default

# Embedding Model
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"

# Chunking
CHUNK_SIZE = 800          # characters per chunk (optimized for Vietnamese)
CHUNK_OVERLAP = 150       # overlap between chunks
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "120"))
ENABLE_CHUNK_DEDUP = os.getenv("ENABLE_CHUNK_DEDUP", "1") == "1"
MAX_CHUNKS_PER_SECTION = int(os.getenv("MAX_CHUNKS_PER_SECTION", "250"))
PDF_MARGIN_REPEAT_RATIO = float(os.getenv("PDF_MARGIN_REPEAT_RATIO", "0.8"))

# Generation context budget
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "2000"))

# Agentic-lite controls
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "0") == "1"
ENABLE_CLARIFICATION_GATE = os.getenv("ENABLE_CLARIFICATION_GATE", "1") == "1"
ENABLE_SELF_CHECK = os.getenv("ENABLE_SELF_CHECK", "0") == "1"

# Search
TOP_K = 3                 # number of results to return
SIMILARITY_THRESHOLD = 40 # minimum similarity % to include in results
RETRIEVAL_CANDIDATE_MULTIPLIER = int(os.getenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "4"))
RERANK_EMBEDDING_WEIGHT = float(os.getenv("RERANK_EMBEDDING_WEIGHT", "0.75"))
RERANK_KEYWORD_WEIGHT = float(os.getenv("RERANK_KEYWORD_WEIGHT", "0.25"))

CLARIFICATION_MIN_TOP_SIMILARITY = float(os.getenv("CLARIFICATION_MIN_TOP_SIMILARITY", "56"))
CLARIFICATION_MARGIN_MIN = float(os.getenv("CLARIFICATION_MARGIN_MIN", "4"))
CLARIFICATION_HIGH_CONFIDENCE = float(os.getenv("CLARIFICATION_HIGH_CONFIDENCE", "80"))
SELF_CHECK_MIN_GROUNDEDNESS = float(os.getenv("SELF_CHECK_MIN_GROUNDEDNESS", "0.35"))

# Generation speed controls
GENERATION_MAX_NEW_TOKENS = int(os.getenv("GENERATION_MAX_NEW_TOKENS", "256"))
REWRITE_MAX_NEW_TOKENS = int(os.getenv("REWRITE_MAX_NEW_TOKENS", "24"))
GENERATION_MAX_TIME_SEC = float(os.getenv("GENERATION_MAX_TIME_SEC", "30"))
FASTEST_RESPONSE_MODE = os.getenv("FASTEST_RESPONSE_MODE", "1") == "1"

# Upload limits
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".txt"}

# Optional API protection for internal deployments
API_KEY = os.getenv("API_KEY", "").strip()

# CORS
CORS_ORIGINS = _parse_csv_env(
	"CORS_ORIGINS",
	["http://localhost:8000", "http://127.0.0.1:8000"]
)

# Server
HOST = "0.0.0.0"
PORT = 8000
