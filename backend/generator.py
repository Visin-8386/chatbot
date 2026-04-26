"""
Generator Service - Local LLM (Qwen2.5-1.5B-Instruct) with float16 on GPU.
No external API needed. Runs entirely on your RTX 3060.
"""
import torch
import threading
import re
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict
from backend.config import (
    MAX_CONTEXT_CHARS,
    GENERATION_MAX_NEW_TOKENS,
    REWRITE_MAX_NEW_TOKENS,
    GENERATION_MAX_TIME_SEC,
)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

_tokenizer = None
_model = None
_tokenizer_lock = threading.Lock()
_model_lock = threading.Lock()

warnings.filterwarnings(
    "ignore",
    message=r"1Torch was not compiled with flash attention.*",
    category=UserWarning,
)


def get_tokenizer() -> AutoTokenizer:
    """Load and cache the tokenizer on first use."""
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                print("Loading tokenizer for local LLM...")
                _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return _tokenizer


def get_model() -> AutoModelForCausalLM:
    """Load and cache the model on first use."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                print("--------------------------------------------------")
                print("Initializing Local LLM (Qwen2.5-1.5B-Instruct) in float16...")
                print("First run will download ~3GB model from HuggingFace.")
                print("Please wait...")
                print("--------------------------------------------------")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                _model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    dtype=torch.float16,
                    device_map=device
                )
                if hasattr(_model, "generation_config"):
                    _model.generation_config.temperature = None
                    _model.generation_config.top_p = None
                    _model.generation_config.top_k = None
                if torch.cuda.is_available():
                    vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    print(f"[GPU] VRAM used: {vram_mb:.0f} MB")
                print("--------------------------------------------------")
                print(f"OK! Local LLM ready on {device.upper()}.")
                print("--------------------------------------------------")
    return _model


def is_model_loaded() -> bool:
    """Return True when the LLM has already been initialized."""
    return _model is not None


def preload_models():
    """Preload tokenizer and model at startup to avoid lock contention during requests."""
    get_tokenizer()
    get_model()


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
    """Run local chat generation with stable defaults."""
    tokenizer = get_tokenizer()
    model = get_model()

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            max_time=GENERATION_MAX_TIME_SEC,
            do_sample=False,
            repetition_penalty=1.1
        )

    output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def rewrite_query(query: str) -> str:
    """Rewrite user query into a concise retrieval-friendly query."""
    clean_query = query.strip()
    if not clean_query:
        return ""

    system_msg = (
        "Bạn là bộ tối ưu truy vấn tìm kiếm tài liệu nội bộ. "
        "Viết lại câu hỏi ngắn gọn, giữ nguyên ý định, bỏ từ dư thừa. "
        "Trả về đúng một câu truy vấn, không giải thích."
    )
    user_msg = f"Truy vấn gốc: {clean_query}"
    rewritten = _run_chat(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        max_new_tokens=REWRITE_MAX_NEW_TOKENS,
        strict=True
    )

    rewritten = rewritten.splitlines()[0].strip() if rewritten else clean_query
    return rewritten or clean_query


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


def generate_answer(query: str, search_results: List[Dict], strict_mode: bool = False) -> Dict:
    """
    Generate answer using local Qwen 2.5 model based on retrieved context chunks.
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

    # Simplified prompt — small models work better with direct instructions
    system_msg = (
        "Bạn là trợ lý AI nội bộ của công ty. "
        "Trả lời câu hỏi dựa trên tài liệu được cung cấp. "
        "Chỉ dùng thông tin có trong tài liệu. "
        "Nếu tài liệu không đủ dữ kiện để kết luận, phải nói rõ 'Không tìm thấy thông tin trong tài liệu được cung cấp'. "
        "Không suy diễn ngoài tài liệu. "
        "Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng."
    )

    user_msg = f"TÀI LIỆU:\n{context_text}\n\nCÂU HỎI: {query}"

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
        print(f"LLM Generation Error: {e}")
        return {
            "answer": "Xin lỗi, đã có lỗi khi tạo câu trả lời. Vui lòng đọc trực tiếp các đoạn văn bản dưới." + citation_text,
            "sources": sources_info
        }


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
