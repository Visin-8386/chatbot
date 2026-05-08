# 🚀 Chiến lược phát triển DocSearch (RAG SOTA Roadmap)

Bản kế hoạch này tập trung vào việc khai thác tối đa sức mạnh của mô hình Local trên hạ tầng **RTX 3060** và các kỹ thuật RAG tiên tiến nhất.

## 📊 Phân tích tài nguyên hiện tại
- **GPU:** RTX 3060 (Ưu tiên các mô hình dưới 8B tham số và kỹ thuật Quantization).
- **LLM:** Qwen2.5-1.5B (Tốc độ cực nhanh, cần nâng cấp lên 7B để có tư duy logic mạnh hơn).
- **Embedding:** e5-small (Tối ưu cho tốc độ).
- **Reranker:** BGE-Reranker-Base (Tiêu chuẩn công nghiệp).

---

## 📍 Giai đoạn 1: Tối ưu hóa độ chính xác nền tảng (Precision & Efficiency)
*Mục tiêu: Đảm bảo tài liệu tìm thấy luôn là tài liệu đúng nhất.*

- [x] **1.1. Semantic Chunking:** Thay thế cách cắt văn bản theo số lượng ký tự bằng cách dùng mô hình Embedding để tìm điểm ngắt câu theo ngữ nghĩa. 
    - *Lợi ích:* Giữ trọn vẹn ý tưởng trong một chunk, không bị mất ngữ cảnh ở giữa câu.
- [x] **1.2. Hybrid Search (Dense + Sparse):** Kết hợp ChromaDB (Vector) với BM25 (Keyword Search).
    - *Lợi ích:* Tìm chính xác các thuật ngữ chuyên môn, mã sản phẩm hoặc tên riêng mà Vector Search đôi khi bỏ lỡ.
- [x] **1.3. LLM Optimization (Quantization):** Chuyển sang sử dụng định dạng GGUF hoặc AWQ cho bản Qwen 7B (Hiện tại đã hỗ trợ NF4 4-bit qua bitsandbytes).
    - *Lợi ích:* Đạt được trình độ suy luận của các model lớn nhưng vẫn chạy mượt trên 12GB VRAM của 3060.

## 🧠 Giai đoạn 2: Nâng cấp trí tuệ hệ thống (Agentic RAG)
*Mục tiêu: Giảm thiểu ảo giác (Hallucination) và xử lý câu hỏi phức tạp.*

- [x] **2.1. Query Expansion (HyDE):** LLM sẽ tạo ra một câu trả lời giả lập trước khi đi tìm kiếm thực tế.
    - *Lợi ích:* Tăng khả năng khớp dữ liệu giữa câu hỏi người dùng và tài liệu chuyên môn.
- [x] **2.2. Self-Correction Loop (CRAG):** Thiết lập bước kiểm tra: "Tài liệu này có thực sự trả lời được câu hỏi không?".
    - *Lợi ích:* Nếu tài liệu rác, LLM sẽ từ chối trả lời thay vì tạo ra thông tin sai lệch.
- [x] **2.3. Persistent Chat Memory:** Lưu lịch sử chat vào database (SQLite) để hỗ trợ hội thoại dài mà không làm tràn Context.

## 🌐 Giai đoạn 3: Hệ thống chuyên nghiệp (Scale & Enterprise)
*Mục tiêu: Đạt chuẩn SOTA và có khả năng ứng dụng thực tế cao.*

- [ ] **3.1. GraphRAG Integration:** Xây dựng bản đồ thực thể (Knowledge Graph) từ tài liệu.
    - *Lợi ích:* Trả lời được các câu hỏi mang tính tổng hợp trên hàng nghìn tài liệu (Ví dụ: "Tổng hợp các thay đổi về chính sách lương trong 5 năm qua").
- [ ] **3.2. Evaluation Suite (Ragas):** Tự động chấm điểm hệ thống dựa trên: Faithfulness (Độ trung thực), Answer Relevance (Độ liên quan), Context Precision (Độ chính xác ngữ cảnh).
- [ ] **3.3. Observability (Trace):** Tích hợp công cụ theo dõi luồng xử lý để biết chính xác mô hình đang "nghĩ" gì ở từng bước.

---
*Lộ trình được thiết kế để triển khai từng bước, mỗi bước hoàn thành sẽ thấy ngay sự thay đổi về chất lượng câu trả lời.*
