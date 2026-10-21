"""
Generator Service - Local LLM (Sailor2-8B-Chat Q4_K_S safe default) via llama-cpp GGUF.
No external API needed. Runs entirely on your RTX 3060 (6GB VRAM budget).
Fallbacks via env: Qwen2.5-7B Q4_K_M / Sailor2 IQ4_XS — see backend/config.py.
"""
import os
import torch
import threading
import re
import warnings
import json
from datetime import datetime
from typing import List, Dict, Generator, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from sentence_transformers import CrossEncoder
from loguru import logger

from backend.config import (
    MAX_CONTEXT_CHARS,
    MAX_HISTORY_TURNS,
    GENERATION_MAX_NEW_TOKENS,
    REWRITE_MAX_NEW_TOKENS,
    GENERATION_MAX_TIME_SEC,
    LLM_MODEL,
    LLM_GGUF_FILENAME,
    LLM_CHAT_FORMAT,
    LLM_N_CTX,
    LLM_N_GPU_LAYERS,
    MODELS_DIR,
    ENABLE_HYDE,
    CRAG_RELEVANCE_THRESHOLD,
)
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

# Use model from config
MODEL_NAME = LLM_MODEL

_tokenizer = None
_model = None
_tokenizer_lock = threading.Lock()
_model_lock = threading.Lock()

warnings.filterwarnings(
    "ignore",
    message=r"1Torch was not compiled with flash attention.*",
    category=UserWarning,
)

# Setup Logger
logger.add("logs/backend.log", rotation="10 MB", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")


def get_tokenizer():
    """Llama-cpp handles tokenization internally."""
    return None


def _preload_torch_models() -> None:
    """Nạp embedding + reranker TRƯỚC llama_cpp (xem chú thích trong get_model)."""
    try:
        from backend.embedding_service import get_model as _get_embedding_model
        _get_embedding_model()
    except Exception as e:
        logger.warning("Embedding preload before LLM failed: {}", e)
    try:
        get_reranker()
    except Exception as e:
        logger.warning("Reranker preload before LLM failed: {}", e)


def get_model() -> Llama:
    """Load and cache the GGUF model on first use (sequential preload — VRAM safe)."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                # Windows + llama_cpp (CUDA): nạp model torch (embedding, reranker)
                # SAU khi llama_cpp active gây access violation native khi đọc
                # safetensors (tái hiện: LLM->emb->rerank crash; emb->rerank->LLM OK).
                # Giữ invariant: mọi torch model load xong TRƯỚC llama_cpp.
                _preload_torch_models()

                logger.info("Downloading/Loading GGUF Model: {} ({})", LLM_MODEL, LLM_GGUF_FILENAME)

                # Download model file if not exists
                model_path = hf_hub_download(
                    repo_id=LLM_MODEL,
                    filename=LLM_GGUF_FILENAME,
                    cache_dir=MODELS_DIR
                )

                logger.info("Initializing Llama-cpp (GGUF) on GPU... n_ctx={} chat_format={}", LLM_N_CTX, LLM_CHAT_FORMAT)
                _model = Llama(
                    model_path=model_path,
                    n_ctx=LLM_N_CTX,          # Keep 4096 on 6GB VRAM
                    n_gpu_layers=LLM_N_GPU_LAYERS,  # -1 = full offload; lower if OOM
                    n_threads=os.cpu_count() or 4,
                    chat_format=LLM_CHAT_FORMAT or None,  # chatml for Sailor2/Qwen2
                    verbose=False
                )

                logger.info("Llama-cpp (GGUF) ready with GPU acceleration.")
    return _model


_reranker = None
_reranker_lock = threading.Lock()

def get_reranker() -> CrossEncoder:
    """Load and cache the reranker model (Vietnamese SOTA default, device-configurable)."""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from backend.config import RERANKER_MODEL, RERANKER_DEVICE, RERANKER_INT8
                logger.info("Loading Reranker: {} (device={})", RERANKER_MODEL, RERANKER_DEVICE)
                if RERANKER_DEVICE == "cuda":
                    device = "cuda"
                elif RERANKER_DEVICE == "cpu":
                    device = "cpu"
                else:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                _reranker = CrossEncoder(RERANKER_MODEL, device=device)  # Cache dir controlled via HF_HOME env var
                if RERANKER_INT8 and device == "cpu":
                    # XLM-R large fp32 trên CPU ~6.5s/9 cặp > SLA 5s; int8 ~3.5s.
                    torch.quantization.quantize_dynamic(
                        _reranker.model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True
                    )
                    logger.info("Reranker quantized to int8 (dynamic, Linear layers).")
                logger.info("Reranker ready on {}.", device.upper())
    return _reranker


def is_model_loaded() -> bool:
    """Return True when the LLM has already been initialized."""
    return _model is not None


def preload_models():
    """Preload LLM and Reranker at startup."""
    get_model()
    get_reranker()


def _build_source_citation_text(sources_info: List[Dict]) -> str:
    """
    Build a formatted citation text block from sources info.
    This is appended to every answer so users ALWAYS know where info came from.
    """
    if not sources_info:
        return ""

    lines = ["\n\n📌 **Nguồn trích dẫn:**"]
    for i, src in enumerate(sources_info):
        parts = [f"- **{src['file']}**"]
        location_parts = []
        if src.get("page"):
            location_parts.append(f"Trang {src['page']}")
        if src.get("sheet"):
            location_parts.append(f"Sheet: {src['sheet']}")
        if src.get("chunk"):
            location_parts.append(f"Đoạn {src['chunk']}")
        if location_parts:
            parts.append(f" ({', '.join(location_parts)})")
        parts.append(f" — Độ liên quan: {src['similarity']}%")
        lines.append("".join(parts))

    return "\n".join(lines)


def _run_chat(messages: List[Dict], max_new_tokens: int = GENERATION_MAX_NEW_TOKENS, strict: bool = False) -> str:
    """Run GGUF chat generation (synchronous). ChatML stop tokens prevent runaway."""
    model = get_model()

    response = model.create_chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.1,
        repeat_penalty=1.1,
        stop=["<|im_start|>", "<|im_end|>"],
    )
    return response["choices"][0]["message"]["content"].strip()


def _run_chat_stream(messages: List[Dict], max_new_tokens: int = GENERATION_MAX_NEW_TOKENS) -> Generator[str, None, None]:
    """Run GGUF chat generation with streaming tokens."""
    model = get_model()

    stream = model.create_chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=0.1,
        repeat_penalty=1.1,
        stop=["<|im_start|>", "<|im_end|>"],
        stream=True
    )

    for chunk in stream:
        if "content" in chunk["choices"][0]["delta"]:
            yield chunk["choices"][0]["delta"]["content"]


def rewrite_query(query: str, history: List[Dict] | None = None) -> str:
    """Rewrite user query into a concise retrieval-friendly query, taking history into account."""
    clean_query = query.strip()
    if not clean_query:
        return ""

    history_text = ""
    if history:
        normalized = _normalize_history(history)
        if normalized:
            lines = []
            for turn in normalized[-2:]:
                if turn.get("user"):
                    lines.append(f"User: {turn['user']}")
                if turn.get("assistant"):
                    short_ans = turn["assistant"].split("\n\n\U0001f4cc")[0][:150]
                    lines.append(f"Bot: {short_ans}")
            if lines:
                history_text = "Lịch sử hội thoại gần nhất:\n" + "\n".join(lines) + "\n\n"

    system_msg = (
        "Bạn là bộ tối ưu truy vấn tìm kiếm tài liệu doanh nghiệp. "
        "Nhiệm vụ: Dựa vào lịch sử hội thoại (nếu có) và câu hỏi mới nhất, hãy viết lại thành MỘT CÂU TRUY VẤN TÌM KIẾM ĐẦY ĐỦ Ý VÀ NGỮ CẢNH. "
        "Ví dụ: Lịch sử 'nghỉ phép 12 ngày/năm', câu mới 'còn gì nữa k' -> Viết lại: 'Các quy định bổ sung về chế độ và thủ tục xin nghỉ phép năm'. "
        "Nghiêm cấm định dạng Markdown (không **in đậm**, không tiêu đề, không bullet), "
        "không tiền tố giải thích ('Truy vấn tìm kiếm:'), không bọc trong ngoặc kép. "
        "Trả về đúng 1 câu truy vấn duy nhất."
    )
    user_msg = f"{history_text}Câu hỏi mới nhất: {clean_query}"
    try:
        max_tokens = REWRITE_MAX_NEW_TOKENS + (16 if history_text else 0)
        rewritten = _run_chat(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_new_tokens=max_tokens,
            strict=True
        )
        cleaned = _clean_rewritten_query(rewritten) if rewritten else ""
        return cleaned or clean_query
    except Exception as e:
        logger.error("Query rewrite error: {}", e)
        return clean_query


def _clean_rewritten_query(text: str) -> str:
    """Lột bỏ Markdown/tiền tố model hay thêm vào câu rewrite.

    Sailor2/Qwen hay trả '**Truy vấn tìm kiếm:** "..."' hoặc '"câu..."'.
    Chỉ giữ nội dung truy vấn thuần để retrieval không bị nhiễu.
    """
    line = (text or "").splitlines()[0].strip() if (text or "").strip() else ""
    if not line:
        return ""
    # Bỏ dấu ** còn sót
    line = line.replace("**", "").strip()
    # Bỏ tiền tố nhãn kiểu "Truy vấn tìm kiếm: <câu>" (model hay thêm)
    label_match = re.match(
        r"^(truy vấn|câu truy vấn|tìm kiếm|câu hỏi viết lại|kết quả)[^:\n]{0,20}:\s*(.+)$",
        line,
        re.IGNORECASE,
    )
    if label_match:
        line = label_match.group(2).strip()
    # Bỏ ngoặc kép bọc ngoài
    if len(line) >= 2 and line[0] == line[-1] and line[0] in ("\"", "'", "“", "”"):
        line = line[1:-1].strip()
    # Bỏ mũi tên giải thích kiểu "cũ -> mới", chỉ lấy vế sau
    if "->" in line:
        line = line.split("->")[-1].strip().strip("\"'“” ")
    return line


def rerank_results(query: str, results: List[Dict], top_n: int = 5) -> List[Dict]:
    """Re-rank retrieved results using a Cross-Encoder for better accuracy."""
    if not results or len(results) <= 1:
        return results

    reranker = get_reranker()

    # Chỉ chấm RERANK_CANDIDATES ứng viên đầu (hybrid đã xếp hạng): chi phí
    # cross-encoder tuyến tính theo số cặp, 18 cặp vượt SLA 5s trên CPU.
    from backend.config import RERANKER_MAX_CHARS, RERANK_CANDIDATES
    results = results[:max(top_n, RERANK_CANDIDATES)]

    # Truncate text: 800 chars giữ đủ trọn chunk luật (đáp án thường nằm cuối
    # điều khoản) — 500 cắt mất; 800 với 6 ứng viên vẫn đạt SLA 5s.
    pairs = [[query, r.get("match_text", r["text"])[:RERANKER_MAX_CHARS]] for r in results]

    scores = reranker.predict(pairs, batch_size=len(pairs), show_progress_bar=False)

    # Chuẩn hóa min-max điểm rerank về 0-100 trong tập ứng viên: CrossEncoder
    # các bản khác nhau trả logit thô hoặc xác suất, và calibration tuyệt đối
    # kém (paraphrase đúng chỉ ~0.06 trong khi exact-match ~0.99) — nhưng thứ tự
    # luôn đúng, nên min-max giữ nguyên ordering mà vẫn phân biệt được.
    raw = [float(s) for s in scores]
    # Guard NaN (cùng lớp lỗi fp32 LayerNorm với embedding): giữ thứ tự hybrid
    # thay vì để NaN phá min-max và làm loạn sort.
    if any(s != s for s in raw):
        logger.warning("Reranker returned NaN scores; skipping rerank, keeping hybrid order.")
        return results[:top_n]
    lo, hi = min(raw), max(raw)
    for i, res in enumerate(results):
        res["rerank_score"] = raw[i]
        if hi > lo:
            rerank_sim = (raw[i] - lo) / (hi - lo) * 100.0
        else:
            rerank_sim = max(0.0, min(1.0, raw[i])) * 100.0
        orig_sim = float(res.get("similarity", 50.0))
        # Fuse scores: 50% Cross-Encoder + 50% Hybrid Vector+BM25 similarity
        res["similarity"] = round(0.5 * rerank_sim + 0.5 * orig_sim, 1)

    results.sort(key=lambda x: (x["similarity"], x["rerank_score"]), reverse=True)
    return results[:top_n]


def build_clarification_question(original_query: str, search_results: List[Dict]) -> str:
    """Generate a concise clarification question when retrieval confidence is low."""
    if not search_results:
        return (
            "Mình chưa đủ dữ liệu để trả lời chính xác. "
            "Bạn có thể nói rõ hơn về phòng ban, mốc thời gian, hoặc tên quy định cần tra cứu không?"
        )

    hints = []
    for res in search_results[:3]:
        source = res.get("metadata", {}).get("source", "Unknown")
        similarity = res.get("similarity", 0)
        hints.append(f"{source} ({similarity}%)")

    hints_text = ", ".join(hints)
    return (
        "Mình cần thêm chi tiết để trả lời chính xác hơn. "
        f"Hiện đang thấy các nguồn gần nhất: {hints_text}. "
        "Bạn muốn hỏi theo quy trình nào cụ thể (ví dụ: nghỉ phép, lương thưởng, làm thêm giờ)?"
    )


def groundedness_score(answer: str, search_results: List[Dict]) -> float:
    """Estimate how much the answer is grounded in retrieved context via lexical overlap."""
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0

    context_text = "\n".join(res.get("text", "") for res in search_results)
    context_tokens = _tokenize(context_text)
    if not context_tokens:
        return 0.0

    return len(answer_tokens.intersection(context_tokens)) / len(answer_tokens)


def _tokenize(text: str) -> set:
    stopwords = {
        "va", "và", "la", "là", "cua", "của", "cho", "tren", "trên", "duoc", "được", "khong", "không",
        "to", "from", "in", "on", "at", "is", "are", "be", "a", "an", "the", "for", "of", "and", "or"
    }
    tokens = re.findall(r"[\w\-]+", text.lower())
    return {token for token in tokens if len(token) > 1 and token not in stopwords}


def _is_follow_up_query(query: str) -> bool:
    """Detect short pronoun-heavy follow-up queries that depend on prior context."""
    normalized = query.strip().lower()
    if not normalized:
        return False
    follow_up_markers = (
        "vay", "vậy", "còn", "con", "thế", "neu", "nếu", "sao", "nhu vay",
        "như vậy", "bo sung", "bổ sung", "tiep", "tiếp"
    )
    token_count = len(_tokenize(normalized))
    marker_hit = any(marker in normalized for marker in follow_up_markers)
    return marker_hit or token_count <= 4


def select_relevant_history(query: str, history: List[Dict] | None) -> List[Dict]:
    """
    Keep history only when the new query is likely a follow-up.
    This prevents old topics from biasing retrieval/generation.
    """
    normalized_history = _normalize_history(history)
    if not normalized_history:
        return []

    if _is_follow_up_query(query):
        return normalized_history

    # Compare topic overlap with the most recent user turn.
    latest_user_query = normalized_history[-1].get("user", "")
    current_tokens = _tokenize(query)
    latest_tokens = _tokenize(latest_user_query)
    if not current_tokens or not latest_tokens:
        return []

    overlap_ratio = len(current_tokens.intersection(latest_tokens)) / len(current_tokens)
    return normalized_history if overlap_ratio >= 0.25 else []


def _prepare_context_and_sources(search_results: List[Dict], max_context_chars: int) -> Dict:
    """Keep the highest-ranked chunks until the context budget is reached."""
    context_parts = []
    sources_info = []
    seen_sources = set()
    used_chars = 0

    for i, res in enumerate(search_results):
        source = res["metadata"].get("source", "Unknown")
        page = res["metadata"].get("page", None)
        sheet = res["metadata"].get("sheet", None)
        chunk_index = res["metadata"].get("chunk_index", None)
        similarity = res.get("similarity", 0)

        source_label = f"Nguồn {i+1}: {source}"
        if page:
            source_label += f" (Trang {page})"
        if sheet:
            source_label += f" (Sheet: {sheet})"

        context_piece = f"[{source_label}]\n{res['text']}"
        projected_chars = used_chars + len(context_piece) + 2
        if context_parts and projected_chars > max_context_chars:
            break

        context_parts.append(context_piece)
        used_chars = projected_chars

        source_key = f"{source}|{page}|{sheet}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            source_entry = {
                "file": source,
                "text": res["text"],
                "similarity": similarity
            }
            if page:
                source_entry["page"] = page
            if sheet:
                source_entry["sheet"] = sheet
            if chunk_index is not None:
                source_entry["chunk"] = int(chunk_index) + 1
            sources_info.append(source_entry)

    return {
        "context_text": "\n\n".join(context_parts),
        "sources_info": sources_info,
    }


def _normalize_history(history: List[Dict] | None) -> List[Dict]:
    """Normalize incoming chat history and keep the most recent turns only."""
    if not history:
        return []
    normalized = []
    for turn in history[-MAX_HISTORY_TURNS:]:
        if not isinstance(turn, dict):
            continue
        user_text = str(turn.get("user", "")).strip()
        assistant_text = str(turn.get("assistant", "")).strip()
        if not user_text and not assistant_text:
            continue
        normalized.append({
            "user": user_text,
            "assistant": assistant_text
        })
    return normalized


def _build_history_text(history: List[Dict]) -> str:
    """Build compact dialogue history text for the prompt."""
    if not history:
        return "Không có hội thoại trước đó."

    lines = []
    for idx, turn in enumerate(history, start=1):
        user_text = turn.get("user", "")
        assistant_text = turn.get("assistant", "")
        lines.append(f"Lượt {idx} - Người dùng: {user_text}")
        if assistant_text:
            lines.append(f"Lượt {idx} - Trợ lý: {assistant_text}")
    return "\n".join(lines)

_skills_cache: Optional[str] = None


def _load_skills() -> str:
    """Tự động nạp tất cả các kỹ năng (skills) từ thư mục skills/.
    
    Kết quả được cache sau lần đọc đầu tiên để tránh I/O disk mỗi lần generate.
    """
    global _skills_cache
    if _skills_cache is not None:
        return _skills_cache

    skills_content = []
    skills_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "skills")
    if os.path.exists(skills_dir):
        for filename in sorted(os.listdir(skills_dir)):
            if filename.endswith(".md"):
                try:
                    with open(os.path.join(skills_dir, filename), "r", encoding="utf-8") as f:
                        skills_content.append(f.read())
                except Exception as e:
                    logger.warning("Không thể nạp skill {}: {}", filename, e)

    _skills_cache = (
        "\n\n--- HƯỚNG DẪN KỸ NĂNG BỔ SUNG ---\n" + "\n\n".join(skills_content)
        if skills_content else ""
    )
    logger.info("Skills cache built: {} file(s) loaded.", len(skills_content))
    return _skills_cache


def generate_answer(query: str, search_results: List[Dict], strict_mode: bool = False, history: List[Dict] | None = None) -> Dict:
    """
    Generate answer using local Sailor2-8B-Chat (Qwen2-based) model based on retrieved context chunks.
    Always appends source citations to the answer.
    
    Returns:
        Dict with 'answer' text (includes citations) and 'sources' list.
    """
    if not search_results:
        return {
            "answer": "Đã tra cứu nhưng không tìm thấy thông tin nào liên quan trong tài liệu.",
            "sources": []
        }

    prepared = _prepare_context_and_sources(search_results, MAX_CONTEXT_CHARS)
    context_text = prepared["context_text"]
    sources_info = prepared["sources_info"]
    normalized_history = _normalize_history(history)
    history_text = _build_history_text(normalized_history)

    # Simplified prompt — small models work better with direct instructions
    system_msg = (
        "Bạn là TRỢ LÝ TRA CỨU TÀI LIỆU DOANH NGHIỆP chuyên nghiệp và nghiêm ngặt (DocSearch nội bộ). "
        "Bạn KHÔNG phải Sailor2/Sea AI Lab/Qwen — không tự xưng tên model gốc, chỉ xưng trợ lý DocSearch. "
        "NHIỆM VỤ: Chỉ cung cấp thông tin CHÍNH XÁC có trong TÀI LIỆU CUNG CẤP. "
        "PHẠM VI: Bạn KHÔNG ĐƯỢC PHÉP sử dụng kiến thức bên ngoài, không giải quyết các vấn đề ngoài lề như lập trình, nấu ăn, hay đời sống. "
        "CÔNG THỨC TRẢ LỜI: Nếu thông tin có trong tài liệu -> Trả lời kèm trích dẫn [Tên_file - Trang]. "
        "Nếu KHÔNG có thông tin -> Trả lời nguyên văn: 'Xin lỗi, tôi là trợ lý tra cứu tài liệu doanh nghiệp. Câu hỏi này nằm ngoài phạm vi tài liệu hiện có nên tôi không thể hỗ trợ.' "
        "Ngôn ngữ: Tiếng Việt, ngắn gọn, trung thực."
    )
    
    # Nạp thêm kỹ năng chuyên gia
    system_msg += _load_skills()

    user_msg = (
        f"HỘI THOẠI GẦN ĐÂY:\n{history_text}\n\n"
        f"TÀI LIỆU:\n{context_text}\n\n"
        f"CÂU HỎI HIỆN TẠI: {query}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    # Build citation text (always appended regardless of LLM output)
    citation_text = _build_source_citation_text(sources_info)

    try:
        if strict_mode:
            messages[0]["content"] += " Ưu tiên trả lời ngắn, liệt kê đúng ý từ tài liệu và tránh diễn giải thêm."

        response = _run_chat(messages, max_new_tokens=GENERATION_MAX_NEW_TOKENS, strict=strict_mode)

        # Always append source citations to the answer
        full_answer = response + citation_text

        return {
            "answer": full_answer,
            "sources": sources_info
        }

    except Exception as e:
        logger.error("LLM Generation Error: {}", e)
        return {
            "answer": "Xin lỗi, đã có lỗi khi tạo câu trả lời. Vui lòng đọc trực tiếp các đoạn văn bản dưới." + citation_text,
            "sources": sources_info
        }


def generate_answer_stream(query: str, search_results: List[Dict], strict_mode: bool = False, history: List[Dict] | None = None) -> Generator[str, None, None]:
    """
    Generate streaming answer using local Sailor2-8B-Chat model.
    Yields JSON chunks for SSE.
    """
    if not search_results:
        yield json.dumps({"answer": "Không tìm thấy thông tin liên quan.", "done": True})
        return

    prepared = _prepare_context_and_sources(search_results, MAX_CONTEXT_CHARS)
    context_text = prepared["context_text"]
    sources_info = prepared["sources_info"]
    normalized_history = _normalize_history(history)
    history_text = _build_history_text(normalized_history)

    system_msg = (
        "Bạn là TRỢ LÝ TRA CỨU TÀI LIỆU DOANH NGHIỆP (DocSearch nội bộ). "
        "Bạn KHÔNG phải Sailor2/Sea AI Lab/Qwen — không tự xưng tên model gốc. "
        "Nếu ngoài phạm vi, hãy từ chối trả lời một cách lịch sự nhưng nghiêm ngặt."
    )
    # Nạp thêm kỹ năng chuyên gia
    system_msg += _load_skills()
    if strict_mode:
        system_msg += " Chỉ trả lời đúng ý chính, không giải thích dài dòng."

    user_msg = f"HỘI THOẠI GẦN ĐÂY:\n{history_text}\n\nTÀI LIỆU:\n{context_text}\n\nCÂU HỎI: {query}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    citation_text = _build_source_citation_text(sources_info)
    
    # Send metadata first
    yield json.dumps({
        "type": "metadata",
        "sources": sources_info,
        "citation_text": citation_text
    }) + "\n"

    try:
        full_response = ""
        for token in _run_chat_stream(messages, max_new_tokens=GENERATION_MAX_NEW_TOKENS):
            full_response += token
            yield json.dumps({"type": "token", "token": token}) + "\n"
        
        yield json.dumps({"type": "done", "full_answer": full_response + citation_text}) + "\n"
    except Exception as e:
        logger.error("LLM Streaming Error: {}", e)
        yield json.dumps({"type": "error", "message": str(e)}) + "\n"


def generate_extractive_answer(search_results: List[Dict]) -> Dict:
    """Return a very fast extractive answer from top-ranked chunks without LLM generation."""
    if not search_results:
        return {
            "answer": "Đã tra cứu nhưng không tìm thấy thông tin nào liên quan trong tài liệu.",
            "sources": []
        }

    prepared = _prepare_context_and_sources(search_results, MAX_CONTEXT_CHARS)
    sources_info = prepared["sources_info"]

    top_text = search_results[0].get("text", "").strip()
    excerpt = top_text[:420].strip()
    if len(top_text) > 420:
        excerpt += "..."

    answer = "Theo tài liệu gần nhất, nội dung liên quan là:\n" + excerpt
    answer += _build_source_citation_text(sources_info)

    return {
        "answer": answer,
        "sources": sources_info
    }


# ─────────────────────────────────────────────────────────────
# Giai đoạn 2.1 — HyDE (Hypothetical Document Embeddings)
# ─────────────────────────────────────────────────────────────

def hyde_expand_query(query: str) -> str:
    """
    Sinh ra một câu trả lời giả lập (hypothetical answer) để dùng làm
    query vector thay vì câu hỏi gốc.

    Lợi ích: Bridging the gap giữa văn phong câu hỏi và văn phong tài liệu,
    giúp vector search tìm được chunk liên quan chính xác hơn.

    Returns:
        Chuỗi: câu trả lời giả lập nếu thành công, ngược lại trả về query gốc.
    """
    if not ENABLE_HYDE:
        return query

    system_msg = (
        "Bạn là chuyên gia tra cứu tài liệu nội bộ công ty. "
        "Hãy viết MỘT đoạn văn ngắn (2-4 câu) mô tả nội dung của tài liệu "
        "mà bạn dự đoán sẽ trả lời được câu hỏi dưới đây. "
        "Chỉ viết đoạn văn mô tả, không giải thích, không thêm tiêu đề."
    )
    user_msg = f"Câu hỏi: {query}"
    try:
        hypothetical = _run_chat(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_new_tokens=120,
        )
        expanded = hypothetical.strip()
        if expanded:
            logger.debug("[HyDE] Expanded query: {}", expanded[:120])
            # Nối câu hỏi gốc + câu trả lời giả lập để giữ cả hai tín hiệu
            return f"{query}\n{expanded}"
    except Exception as e:
        logger.warning("[HyDE] Failed, falling back to original query: {}", e)
    return query


# ─────────────────────────────────────────────────────────────
# Giai đoạn 2.2 — CRAG (Corrective RAG Relevance Gate)
# ─────────────────────────────────────────────────────────────

def crag_check_relevance(query: str, search_results: List[Dict]) -> Dict:
    """
    Kiểm tra mức độ liên quan của các chunk đã lấy về so với câu hỏi.

    Ba kết quả có thể xảy ra:
      - "relevant":   Tài liệu đủ tốt → tiếp tục generate.
      - "ambiguous":  Tài liệu khớp một phần → generate nhưng thêm disclaimer.
      - "irrelevant": Tài liệu không liên quan → từ chối generate, báo người dùng.

    Returns:
        {"verdict": str, "reason": str, "filtered_results": List[Dict]}
    """
    if not search_results:
        return {
            "verdict": "irrelevant",
            "reason": "Không tìm thấy tài liệu nào liên quan.",
            "filtered_results": []
        }

    top_similarity = search_results[0].get("similarity", 0)

    # Fast-path: dựa vào similarity score để tránh tốn token LLM
    if top_similarity >= CRAG_RELEVANCE_THRESHOLD:
        return {
            "verdict": "relevant",
            "reason": f"Top similarity {top_similarity:.1f}% vượt ngưỡng.",
            "filtered_results": search_results
        }

    if top_similarity < CRAG_RELEVANCE_THRESHOLD * 0.5:
        return {
            "verdict": "irrelevant",
            "reason": f"Top similarity {top_similarity:.1f}% quá thấp. Không đủ dữ liệu để trả lời.",
            "filtered_results": []
        }

    # Vùng mờ: nhờ LLM phán xét nhanh
    context_preview = search_results[0]["text"][:400]
    system_msg = (
        "Trả lời đúng MỘT từ: RELEVANT, AMBIGUOUS, hoặc IRRELEVANT.\n"
        "RELEVANT    = tài liệu có thể trả lời trực tiếp câu hỏi.\n"
        "AMBIGUOUS   = tài liệu liên quan một phần nhưng không đủ.\n"
        "IRRELEVANT  = tài liệu hoàn toàn không liên quan."
    )
    user_msg = (
        f"Câu hỏi: {query}\n\n"
        f"Đoạn tài liệu:\n{context_preview}"
    )
    verdict = "ambiguous"  # default
    try:
        raw = _run_chat(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ],
            max_new_tokens=8,
        ).strip().upper()
        if "RELEVANT" in raw and "IRRELEVANT" not in raw:
            verdict = "relevant"
        elif "IRRELEVANT" in raw:
            verdict = "irrelevant"
        else:
            verdict = "ambiguous"
    except Exception as e:
        logger.warning("[CRAG] LLM check failed: {}", e)

    logger.info("[CRAG] verdict={} sim={:.1f}%", verdict, top_similarity)
    return {
        "verdict": verdict,
        "reason": f"LLM verdict: {verdict} (similarity {top_similarity:.1f}%).",
        "filtered_results": search_results if verdict != "irrelevant" else []
    }
