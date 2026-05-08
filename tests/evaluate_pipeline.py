"""
RAG Evaluation Suite (LLM-as-a-Judge)

Đánh giá chất lượng của pipeline dựa trên 3 metric chính (mô phỏng framework Ragas):
1. Faithfulness (Độ trung thực): Câu trả lời có bị ảo giác (hallucinate) so với context không?
2. Answer Relevance (Độ liên quan): Câu trả lời có trả lời đúng trọng tâm câu hỏi không?
3. Context Relevance (Độ chính xác ngữ cảnh): Các chunk lấy lên có chứa thông tin trả lời không?

Sử dụng chính mô hình Qwen 2.5 làm "Giám khảo" (Judge) để chấm điểm từ 0-10.
"""
import os
import sys
import json
import asyncio
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from backend.vector_store import search
from backend.generator import generate_answer, get_model, rerank_results
from backend.config import ENABLE_CONTEXT_EXPANSION

EVAL_DATASET = [
    {
        "query": "Quy trình xin nghỉ phép hàng năm như thế nào?",
        "context_hint": "Nhân viên được nghỉ 12 ngày/năm. Đơn phải nộp trước 3 ngày."
    },
    {
        "query": "Mã thiết bị MAINT-2024-SERVER được đặt ở đâu?",
        "context_hint": "MAINT-2024-SERVER tại DC-01."
    },
    {
        "query": "Công ty có hỗ trợ ăn trưa không?",
        "context_hint": "Thông tin không có trong tài liệu."
    }
]

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


def _get_score_from_llm(prompt: str) -> int:
    """Gọi LLM để lấy điểm số từ 0-10."""
    try:
        model = get_model()
        messages = [{"role": "user", "content": prompt.strip()}]
        response = model.create_chat_completion(
            messages=messages,
            max_tokens=5,
            temperature=0.0,
        )
        text = response["choices"][0]["message"]["content"].strip()
        # Extract number
        nums = [int(s) for s in text.split() if s.isdigit()]
        if nums:
            return min(10, max(0, nums[0]))
        return 0
    except Exception as e:
        logger.error(f"Lỗi chấm điểm: {e}")
        return 0


async def evaluate_single_query(query: str) -> dict:
    """Chạy pipeline và chấm điểm cho 1 query."""
    logger.info(f"Evaluating: {query}")
    
    # 1. Retrieval
    results = search(query, top_k=3)
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
    
    return {
        "query": query,
        "answer": plain_answer,
        "faithfulness": faithfulness_score,
        "relevance": relevance_score,
    }


async def run_evaluation_suite():
    print("="*60)
    print(" RAG EVALUATION SUITE (LLM-as-a-Judge)")
    print("="*60)
    
    total_f = 0
    total_r = 0
    
    results = []
    
    for item in EVAL_DATASET:
        res = await evaluate_single_query(item["query"])
        results.append(res)
        total_f += res["faithfulness"]
        total_r += res["relevance"]
        
        print(f"\nQ: {res['query']}")
        print(f"A: {res['answer'][:100]}...")
        print(f"Scores -> Faithfulness: {res['faithfulness']}/10 | Relevance: {res['relevance']}/10")
    
    avg_f = total_f / len(EVAL_DATASET)
    avg_r = total_r / len(EVAL_DATASET)
    
    print("\n" + "="*60)
    print(f" KẾT QUẢ TỔNG QUAN")
    print("="*60)
    print(f"Trung bình Faithfulness (Không ảo giác): {avg_f:.1f}/10")
    print(f"Trung bình Answer Relevance (Đúng trọng tâm): {avg_r:.1f}/10")
    print("="*60)
    
    # Save report
    report_path = "eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Đã lưu report chi tiết tại {report_path}")


if __name__ == "__main__":
    asyncio.run(run_evaluation_suite())
