"""
RAG Evaluation Suite (LLM-as-a-Judge, offline RAGAS-style)

Đánh giá chất lượng pipeline dựa trên 3 metric chính (mô phỏng framework Ragas,
chạy hoàn toàn offline bằng chính LLM local — không gọi API ngoài):
1. Faithfulness (Độ trung thực): Câu trả lời có bị ảo giác (hallucinate) so với context không?
2. Answer Relevance (Độ liên quan): Câu trả lời có trả lời đúng trọng tâm câu hỏi không?
3. Context Precision (Độ chính xác ngữ cảnh): Các chunk lấy lên có chứa thông tin trả lời không?

Sử dụng chính mô hình local (Sailor2-8B/Qwen) làm "Giám khảo" (Judge) để chấm điểm từ 0-10.
Chạy: python tests/evaluate_pipeline.py  (yêu cầu server/model đã load)
Kết quả: eval_report.json (schema: query/answer/faithfulness/relevance/context_precision)
"""
import os
import sys
import json
import asyncio
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Nạp .env như main.py để eval chạy đúng cấu hình production
# (RERANKER_DEVICE, EMBEDDING_DEVICE, ...) — không override env shell.
from dotenv import load_dotenv

load_dotenv(override=False)

from loguru import logger
from backend.vector_store import search
from backend.generator import generate_answer, get_model, rerank_results
from backend.config import ENABLE_CONTEXT_EXPANSION, RETRIEVAL_CANDIDATE_MULTIPLIER

# Cho phép chạy trên bộ câu hỏi khác (vd. corpus luật) mà không sửa code:
#   EVAL_DATASET_FILE=tests/legal_eval_queries.json EVAL_REPORT_FILE=eval_legal.json python tests/evaluate_pipeline.py
EVAL_DATASET_FILE = os.getenv("EVAL_DATASET_FILE", "").strip()
EVAL_REPORT_FILE = os.getenv("EVAL_REPORT_FILE", "eval_report.json").strip() or "eval_report.json"

EVAL_DATASET = [
    # Factoid nghỉ phép (single-hop)
    {
        "query": "Quy trình xin nghỉ phép hàng năm như thế nào?",
        "context_hint": "Nhân viên được nghỉ 12 ngày/năm. Đơn phải nộp trước 3 ngày."
    },
    # Mã thiết bị chính xác (keyword-sensitive: BM25 phải bắt được)
    {
        "query": "Mã thiết bị MAINT-2024-SERVER được đặt ở đâu?",
        "context_hint": "MAINT-2024-SERVER tại DC-01."
    },
    # Out-of-scope -> hệ thống phải từ chối, không bịa
    {
        "query": "Công ty có hỗ trợ ăn trưa không?",
        "context_hint": "Thông tin không có trong tài liệu."
    },
    # Bảng biểu (table-aware chunking + parser)
    {
        "query": "Mức phụ cấp trong bảng lương cho nhân viên thử việc là bao nhiêu?",
        "context_hint": "Bảng lương: thử việc 85% + phụ cấp theo bảng."
    },
    # Follow-up cần history ("còn gì nữa")
    {
        "query": "Còn quy định bổ sung nào về nghỉ phép nữa không?",
        "context_hint": "Quy định bổ sung về nghỉ phép: phép bù, phép không lương."
    },
    # Câu hỏi thiếu ngữ cảnh -> clarification gate
    {
        "query": "Quy định đó áp dụng từ khi nào?",
        "context_hint": "Mơ hồ nếu thiếu history; cần hỏi lại mốc thời gian/văn bản."
    },
    # Sheet Excel (sheet metadata + citation)
    {
        "query": "Tồn kho linh kiện tháng 3 ở sheet nào, số lượng bao nhiêu?",
        "context_hint": "Sheet TonKho-T3: mã linh kiện + số lượng tồn."
    },
    # Tổng hợp đa tài liệu (global query — hướng GraphRAG)
    {
        "query": "Tổng hợp các thay đổi về chính sách lương trong các văn bản hiện có?",
        "context_hint": "Cần nhiều chunk từ nhiều văn bản; đo khả năng tổng hợp."
    },
    # Hỏi ngoài phạm vi kiến thức chung (lập trình/nấu ăn) -> phải từ chối
    {
        "query": "Viết giúp tôi hàm quicksort bằng Python?",
        "context_hint": "Ngoài phạm vi tài liệu doanh nghiệp; phải từ chối."
    },
    # Câu chào hỏi (intent chitchat, không vào RAG)
    {
        "query": "Xin chào, bạn là ai?",
        "context_hint": "Chitchat; trả lời giới thiệu, không truy xuất tài liệu."
    },
    # Số liệu + ngày tháng chính xác (chống ảo giác số)
    {
        "query": "Hạn nộp báo cáo quý là ngày nào, nộp cho ai?",
        "context_hint": "Hạn nộp + đầu mối nhận báo cáo phải khớp nguyên văn tài liệu."
    },
    # Viết tắt/thuật ngữ nội bộ (hybrid search: dense + BM25)
    {
        "query": "PCCC là gì và ai chịu trách nhiệm kiểm tra định kỳ?",
        "context_hint": "Thuật ngữ viết tắt nội bộ; cần keyword match + giải thích."
    },
]

# Nạp dataset thay thế từ file JSON (list các {"query", "context_hint"}).
if EVAL_DATASET_FILE and os.path.exists(EVAL_DATASET_FILE):
    with open(EVAL_DATASET_FILE, encoding="utf-8") as _f:
        EVAL_DATASET = json.load(_f)
    print(f"Loaded {len(EVAL_DATASET)} queries from {EVAL_DATASET_FILE}")

JUDGE_PROMPT_FAITHFULNESS = """
Bạn là một giám khảo chấm điểm khắt khe. Nhiệm vụ của bạn là đánh giá "Độ trung thực" (Faithfulness) của câu trả lời so với Tài liệu tham khảo.
Quy tắc:
- Trả về ĐIỂM từ 0 đến 10.
- Nếu câu trả lời chứa thông tin KHÔNG CÓ trong tài liệu tham khảo (bịa đặt), điểm phải rất thấp (0-3).
- Nếu câu trả lời hoàn toàn dựa vào tài liệu tham khảo, điểm cao (8-10).
- Chỉ trả về một số nguyên từ 0-10. Không giải thích gì thêm.

TÀI LIỆU THAM KHẢO:
{context}

CÂU TRẢ LỜI CỦA HỆ THỐNG:
{answer}

ĐIỂM FAITHFULNESS (0-10):
"""

JUDGE_PROMPT_RELEVANCE = """
Bạn là một giám khảo chấm điểm khắt khe. Nhiệm vụ của bạn là đánh giá "Độ liên quan" (Answer Relevance) của câu trả lời so với Câu hỏi.
Quy tắc:
- Trả về ĐIỂM từ 0 đến 10.
- Nếu câu trả lời trả lời đúng và trực tiếp vào câu hỏi, điểm cao (8-10).
- Nếu câu trả lời lảng tránh, lạc đề hoặc nói "Tôi không biết" dù có thể trả lời, điểm thấp (0-4).
- Nếu hệ thống trả lời "Không có thông tin" vì tài liệu thực sự không có, và đó là hành động đúng, hãy cho điểm 10.
- Chỉ trả về một số nguyên từ 0-10. Không giải thích gì thêm.

CÂU HỎI CỦA NGƯỜI DÙNG:
{query}

CÂU TRẢ LỜI CỦA HỆ THỐNG:
{answer}

ĐIỂM RELEVANCE (0-10):
"""

JUDGE_PROMPT_CONTEXT_PRECISION = """
Bạn là một giám khảo chấm điểm khắt khe. Nhiệm vụ của bạn là đánh giá "Độ chính xác ngữ cảnh" (Context Precision): các đoạn tài liệu truy xuất được có chứa thông tin để trả lời câu hỏi không?
Quy tắc:
- Trả về ĐIỂM từ 0 đến 10.
- Nếu đoạn tài liệu chứa trực tiếp câu trả lời (số liệu, tên, quy trình đúng trọng tâm), điểm cao (8-10).
- Nếu chỉ liên quan một phần / chung chung, điểm trung bình (4-7).
- Nếu hoàn toàn lạc đề hoặc rỗng, điểm thấp (0-3).
- Chỉ trả về một số nguyên từ 0-10. Không giải thích gì thêm.

CÂU HỎI CỦA NGƯỜI DÙNG:
{query}

CÁC ĐOẠN TÀI LIỆU TRUY XUẤT:
{context}

ĐIỂM CONTEXT PRECISION (0-10):
"""


def _get_score_from_llm(prompt: str) -> int:
    """Gọi LLM để lấy điểm số từ 0-10."""
    import re
    try:
        model = get_model()
        messages = [{"role": "user", "content": prompt.strip()}]
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=8,
            temperature=0.0,
        )
        text = response["choices"][0]["message"]["content"].strip()
        # Extract number: model thường trả "**8**", "8.", "Điểm: 8" — không phải
        # token thuần số nên không dùng str.isdigit().
        m = re.search(r"\d+", text)
        if m:
            return min(10, max(0, int(m.group())))
        return 0
    except Exception as e:
        logger.error(f"Lỗi chấm điểm: {e}")
        return 0


async def evaluate_single_query(query: str) -> dict:
    """Chạy pipeline và chấm điểm cho 1 query."""
    logger.info(f"Evaluating: {query}")
    
    # 1. Retrieval — pool giống production (top_k * RETRIEVAL_CANDIDATE_MULTIPLIER)
    results = search(query, top_k=3 * RETRIEVAL_CANDIDATE_MULTIPLIER)
    results = rerank_results(query, results, top_n=3)
    
    context_text = "\n---\n".join([r["text"] for r in results])
    
    # 2. Generation
    ai_result = generate_answer(query, results)
    answer = ai_result["answer"]
    
    # Lược bỏ phần Nguồn trích dẫn để judge khách quan
    plain_answer = answer.split("\n\n📌")[0].strip()
    
    # 3. Chấm điểm Faithfulness (Độ trung thực)
    f_prompt = JUDGE_PROMPT_FAITHFULNESS.format(context=context_text, answer=plain_answer)
    faithfulness_score = _get_score_from_llm(f_prompt)
    
    # 4. Chấm điểm Relevance (Độ liên quan)
    r_prompt = JUDGE_PROMPT_RELEVANCE.format(query=query, answer=plain_answer)
    relevance_score = _get_score_from_llm(r_prompt)

    # 5. Chấm điểm Context Precision (RAGAS-style: retrieval có trúng không?)
    c_prompt = JUDGE_PROMPT_CONTEXT_PRECISION.format(query=query, context=context_text)
    context_precision_score = _get_score_from_llm(c_prompt)

    return {
        "query": query,
        "answer": plain_answer,
        "faithfulness": faithfulness_score,
        "relevance": relevance_score,
        "context_precision": context_precision_score,
    }


async def run_evaluation_suite():
    print("="*60)
    print(" RAG EVALUATION SUITE (LLM-as-a-Judge)")
    print("="*60)
    
    total_f = 0
    total_r = 0
    total_c = 0

    results = []

    for item in EVAL_DATASET:
        res = await evaluate_single_query(item["query"])
        results.append(res)
        total_f += res["faithfulness"]
        total_r += res["relevance"]
        total_c += res["context_precision"]

        print(f"\nQ: {res['query']}")
        print(f"A: {res['answer'][:100]}...")
        print(f"Scores -> Faithfulness: {res['faithfulness']}/10 | Relevance: {res['relevance']}/10 | ContextPrecision: {res['context_precision']}/10")

    avg_f = total_f / len(EVAL_DATASET)
    avg_r = total_r / len(EVAL_DATASET)
    avg_c = total_c / len(EVAL_DATASET)

    print("\n" + "="*60)
    print(f" KẾT QUẢ TỔNG QUAN ({len(EVAL_DATASET)} queries)")
    print("="*60)
    print(f"Trung bình Faithfulness (Không ảo giác): {avg_f:.1f}/10")
    print(f"Trung bình Answer Relevance (Đúng trọng tâm): {avg_r:.1f}/10")
    print(f"Trung bình Context Precision (Truy xuất trúng): {avg_c:.1f}/10")
    print("="*60)
    print(f"Trung bình Faithfulness (Không ảo giác): {avg_f:.1f}/10")
    print(f"Trung bình Answer Relevance (Đúng trọng tâm): {avg_r:.1f}/10")
    print("="*60)
    
    # Save report
    report_path = EVAL_REPORT_FILE
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Đã lưu report chi tiết tại {report_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation_suite())
