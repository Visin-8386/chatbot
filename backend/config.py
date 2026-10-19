"""
Configuration for the Document Chatbot system.
"""
import os
from typing import List

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# Model Storage (Keep everything in the project directory)
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Set HuggingFace Cache Directory
os.environ["HF_HOME"] = MODELS_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODELS_DIR


def _parse_csv_env(name: str, default: List[str]) -> List[str]:
	raw_value = os.getenv(name, "").strip()
	if not raw_value:
		return default

	values = [item.strip() for item in raw_value.split(",") if item.strip()]
	return values or default

# Embedding Model (SOTA Vietnamese default; free + offline after first download)
# AITeamVN/Vietnamese_Embedding_v2: fine-tuned from bge-m3, 1024 dims, Apache-2.0.
# Rollback via env: EMBEDDING_MODEL=intfloat/multilingual-e5-small (384 dims).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "AITeamVN/Vietnamese_Embedding_v2")
# E5-family needs "query:/passage:" prefixes; bge-m3-based Vietnamese models do not.
# Auto-detect: enabled only when model name contains "e5" unless explicitly overridden.
_EMBEDDING_PREFIX_ENV = os.getenv("EMBEDDING_USE_E5_PREFIX", "").strip()
EMBEDDING_USE_E5_PREFIX = (
    _EMBEDDING_PREFIX_ENV == "1" if _EMBEDDING_PREFIX_ENV
    else "e5" in EMBEDDING_MODEL.lower()
)
# Vector collection is versioned per embedding model so a dimension change
# (384 -> 1024) never corrupts the old index. Pin via VECTOR_COLLECTION to rollback.
VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "").strip()  # empty = auto from EMBEDDING_MODEL

# LLM Model (GGUF Version for Speed & Efficiency)
# SOTA safe default: Sailor2-8B-Chat Q4_K_S (~4.99GB) — SEA/Vietnamese-tuned,
# fits RTX 3060 6GB with n_ctx=4096. Override via env for rollback or fallback:
#   Qwen2.5-7B Q4_K_M (lighter) / Sailor2 IQ4_XS (lightest, ~4.71GB).
LLM_MODEL = os.getenv("LLM_MODEL", "bartowski/Sailor2-8B-Chat-GGUF")
LLM_GGUF_FILENAME = os.getenv("LLM_GGUF_FILENAME", "Sailor2-8B-Chat-Q4_K_S.gguf")
LLM_CHAT_FORMAT = os.getenv("LLM_CHAT_FORMAT", "chatml")  # Sailor2/Qwen2 use ChatML
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "4096"))  # Keep 4096 on 6GB VRAM; do not raise without measuring
LLM_N_GPU_LAYERS = int(os.getenv("LLM_N_GPU_LAYERS", "-1"))  # -1 = full offload; use 0 or fewer layers if OOM
LLM_LOAD_IN_4BIT = True # GGUF is already quantized

# Chunking
CHUNK_SIZE = 800          # characters per chunk (optimized for Vietnamese)
CHUNK_OVERLAP = 150       # overlap between chunks
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "120"))
ENABLE_CHUNK_DEDUP = os.getenv("ENABLE_CHUNK_DEDUP", "1") == "1"
MAX_CHUNKS_PER_SECTION = int(os.getenv("MAX_CHUNKS_PER_SECTION", "2000"))
PDF_MARGIN_REPEAT_RATIO = float(os.getenv("PDF_MARGIN_REPEAT_RATIO", "0.8"))

# Generation context budget
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "8000"))  # Tăng để lấy nhiều nội dung hơn từ tài liệu
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))

# Agentic-lite controls
ENABLE_QUERY_REWRITE = os.getenv("ENABLE_QUERY_REWRITE", "0") == "1"
ENABLE_CLARIFICATION_GATE = os.getenv("ENABLE_CLARIFICATION_GATE", "0") == "1"
ENABLE_SELF_CHECK = os.getenv("ENABLE_SELF_CHECK", "0") == "1"

# Giai đoạn 2: Agentic RAG
ENABLE_HYDE = os.getenv("ENABLE_HYDE", "0") == "1"                       # 2.1 Query Expansion
CRAG_RELEVANCE_THRESHOLD = float(os.getenv("CRAG_RELEVANCE_THRESHOLD", "30"))  # 2.2 CRAG gate (%)
ENABLE_PERSISTENT_MEMORY = os.getenv("ENABLE_PERSISTENT_MEMORY", "1") == "1"   # 2.3 SQLite memory

# Hybrid fusion: "weighted" (mặc định SOTA: α-interpolation dense+BM25) hoặc "rrf".
HYBRID_FUSION = os.getenv("HYBRID_FUSION", "weighted").strip().lower()
HYBRID_DENSE_WEIGHT = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.7"))  # trọng số dense (0..1)
HYBRID_RRF_K = int(os.getenv("HYBRID_RRF_K", "60"))  # hằng số RRF khi HYBRID_FUSION=rrf

# Search
TOP_K = int(os.getenv("TOP_K", "6"))                 # number of results to return
ENABLE_CONTEXT_EXPANSION = os.getenv("ENABLE_CONTEXT_EXPANSION", "1") == "1" # Tự động lấy chunk trước/sau
SIMILARITY_THRESHOLD = 30 # minimum similarity % to include in results
RETRIEVAL_CANDIDATE_MULTIPLIER = int(os.getenv("RETRIEVAL_CANDIDATE_MULTIPLIER", "3"))
RERANK_EMBEDDING_WEIGHT = float(os.getenv("RERANK_EMBEDDING_WEIGHT", "0.75"))
RERANK_KEYWORD_WEIGHT = float(os.getenv("RERANK_KEYWORD_WEIGHT", "0.25"))
# Reranker (SOTA Vietnamese default; free + offline after first download)
# AITeamVN/Vietnamese_Reranker: fine-tuned from bge-reranker-v2-m3, 0.6B, Apache-2.0.
# Rollback via env: RERANKER_MODEL=BAAI/bge-reranker-base (lighter, 278M).
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "AITeamVN/Vietnamese_Reranker")
# RERANKER_DEVICE=auto | cuda | cpu. "cpu" frees VRAM for the LLM on 6GB cards
# (rerank of ~12 short pairs is fast enough on CPU).
_RERANKER_DEVICE_ENV = os.getenv("RERANKER_DEVICE", "auto").strip().lower()
RERANKER_DEVICE = _RERANKER_DEVICE_ENV if _RERANKER_DEVICE_ENV in ("auto", "cuda", "cpu") else "auto"
# EMBEDDING_DEVICE=cpu | cuda | auto. Default "cpu": Vietnamese embedding 0.6B
# (~2.4GB fp32) would otherwise crowd the 6GB VRAM next to the 5GB LLM.
# CPU inference (~100-300ms/query) is fast enough and keeps GPU for generation.
_EMBEDDING_DEVICE_ENV = os.getenv("EMBEDDING_DEVICE", "cpu").strip().lower()
EMBEDDING_DEVICE = _EMBEDDING_DEVICE_ENV if _EMBEDDING_DEVICE_ENV in ("auto", "cuda", "cpu") else "cpu"
# Model card AITeamVN/Vietnamese_Embedding_v2: huấn luyện query 256 / passage 2048
# token — cap 2048 tránh nổ activation với chunk khổng lồ (từng OOM 7.4GB VRAM).
EMBEDDING_MAX_SEQ_LENGTH = int(os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "2048"))
# Truncate chunk text khi rerank: 500 chars — đủ trọn chunk luật (~650 ký tự)
# để cross-encoder thấy cả điều khoản nằm cuối chunk (300 cắt mất đáp án "40 giờ").
RERANKER_MAX_CHARS = int(os.getenv("RERANKER_MAX_CHARS", "500"))

# Clarification Gate Thresholds
# Lưu ý: trên corpus lớn, các đoạn boilerplate trùng nhau làm top-score thấp giả
# và margin = 0; ngưỡng cũ (25 / 0.5) hỏi lại cả câu hỏi rõ ràng. search() đã
# dedupe text trùng nên margin giờ mới phản ánh khác biệt thực.
CLARIFICATION_MIN_TOP_SIMILARITY = float(os.getenv("CLARIFICATION_MIN_TOP_SIMILARITY", "18"))
CLARIFICATION_MARGIN_MIN = float(os.getenv("CLARIFICATION_MARGIN_MIN", "3.0"))
CLARIFICATION_HIGH_CONFIDENCE = float(os.getenv("CLARIFICATION_HIGH_CONFIDENCE", "40"))
SELF_CHECK_MIN_GROUNDEDNESS = float(os.getenv("SELF_CHECK_MIN_GROUNDEDNESS", "0.35"))

# Generation speed controls
GENERATION_MAX_NEW_TOKENS = int(os.getenv("GENERATION_MAX_NEW_TOKENS", "512"))  # Tăng để trả lời đầy đủ hơn
REWRITE_MAX_NEW_TOKENS = int(os.getenv("REWRITE_MAX_NEW_TOKENS", "24"))
GENERATION_MAX_TIME_SEC = float(os.getenv("GENERATION_MAX_TIME_SEC", "30"))
FASTEST_RESPONSE_MODE = os.getenv("FASTEST_RESPONSE_MODE", "0") == "1"

# Document parsing (SOTA: Docling CPU when installed, PyPDF2 fallback otherwise)
# DOC_PARSER=auto | pypdf | docling. "auto" uses Docling only if importable.
DOC_PARSER = os.getenv("DOC_PARSER", "auto").strip().lower()
# Docling runs CPU-only by default (no VRAM taken from the LLM).
DOCLING_ARTIFACTS_PATH = os.getenv("DOCLING_ARTIFACTS_PATH", os.path.join(MODELS_DIR, "docling")).strip()

# Giai đoạn 3.1: GraphRAG-lite (Knowledge Graph SQLite + 1-hop expansion)
# Mở opt-in: cần upload lại tài liệu sau khi bật để index graph.
ENABLE_GRAPH_RAG = os.getenv("ENABLE_GRAPH_RAG", "0") == "1"
GRAPH_MAX_EXPAND = int(os.getenv("GRAPH_MAX_EXPAND", "3"))  # chunk graph tối đa thêm vào kết quả
GRAPH_USE_LLM = os.getenv("GRAPH_USE_LLM", "0") == "1"     # trích triple bằng LLM local (tốn 1 call/chunk lúc ingest)

# Upload limits
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Supported file types
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".txt"}

# Optional API protection for internal deployments
API_KEY = os.getenv("API_KEY", "").strip()
# RBAC-lite: phân quyền read/write bằng key riêng (CSV). API_KEY luôn full-quyền.
API_READ_KEYS = os.getenv("API_READ_KEYS", "").strip()
API_WRITE_KEYS = os.getenv("API_WRITE_KEYS", "").strip()

# Async ingest (in-process background jobs; không cần Redis)
# Mặc định 0 = upload đồng bộ như cũ. Bật 1 khi parse file nặng (Docling/OCR).
ENABLE_ASYNC_INGEST = os.getenv("ENABLE_ASYNC_INGEST", "0") == "1"

# CORS
CORS_ORIGINS = _parse_csv_env(
	"CORS_ORIGINS",
	["http://localhost:8000", "http://127.0.0.1:8000"]
)

# Server
HOST = "0.0.0.0"
PORT = 8000

# Rate Limiting (per IP, in-memory sliding window)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SEC = int(os.getenv("RATE_LIMIT_WINDOW_SEC", "60"))
