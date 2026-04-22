"""
Generator Service - Local LLM (Qwen2.5-1.5B-Instruct) with float16 on GPU.
No external API needed. Runs entirely on your RTX 3060.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict

print("--------------------------------------------------")
print("Initializing Local LLM (Qwen2.5-1.5B-Instruct) in float16...")
print("First run will download ~3GB model from HuggingFace.")
print("Please wait...")
print("--------------------------------------------------")

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load model in float16 onto GPU (~3GB VRAM, fits easily on 6GB RTX 3060)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("--------------------------------------------------")
print("OK! Local LLM ready. Model loaded on:", model.device)
print("--------------------------------------------------")


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


def generate_answer(query: str, search_results: List[Dict]) -> Dict:
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

    # Build context from search results and collect source info
    context_parts = []
    sources_info = []
    seen_sources = set()

    for i, res in enumerate(search_results):
        source = res["metadata"].get("source", "Unknown")
        page = res["metadata"].get("page", None)
        sheet = res["metadata"].get("sheet", None)
        chunk_index = res["metadata"].get("chunk_index", None)
        similarity = res.get("similarity", 0)

        # Build source label for context
        source_label = f"Nguồn {i+1}: {source}"
        if page:
            source_label += f" (Trang {page})"
        if sheet:
            source_label += f" (Sheet: {sheet})"

        context_parts.append(f"[{source_label}]\n{res['text']}")

        # Collect unique sources for citation
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

    context_text = "\n\n".join(context_parts)

    # Simplified prompt — small models work better with direct instructions
    system_msg = (
        "Bạn là trợ lý AI nội bộ của công ty. "
        "Trả lời câu hỏi dựa trên tài liệu được cung cấp. "
        "Chỉ dùng thông tin có trong tài liệu. "
        "Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng."
    )

    user_msg = f"TÀI LIỆU:\n{context_text}\n\nCÂU HỎI: {query}"

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg}
    ]

    # Build citation text (always appended regardless of LLM output)
    citation_text = _build_source_citation_text(sources_info)

    import time

    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                repetition_penalty=1.1,
                top_p=0.9
            )
        end_time = time.time()

        # Strip input tokens from output
        output_ids = generated_ids[0][model_inputs.input_ids.shape[1]:]
        
        # Calculate Tokens Per Second (TPS)
        num_tokens = len(output_ids)
        generation_time = end_time - start_time
        tps = num_tokens / generation_time if generation_time > 0 else 0
        print(f"✅ LLM Speed: {num_tokens} tokens in {generation_time:.2f}s ({tps:.2f} tokens/sec)")

        response = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

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
