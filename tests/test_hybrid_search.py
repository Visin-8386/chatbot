import os
import sys
from loguru import logger

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.vector_store import add_documents, search, get_collection
from backend.embedding_service import get_model

def test_hybrid_retrieval():
    logger.info("Starting Hybrid Search Test...")
    
    # 1. Clear or ensure collection
    collection = get_collection()
    
    # 2. Add test documents
    test_docs = [
        {
            "text": "Quy trình vận hành máy chủ tại trung tâm dữ liệu ACME. Mã số bảo trì: MAINT-2024-SERVER.",
            "metadata": {"source": "test1.txt"}
        },
        {
            "text": "Hướng dẫn sử dụng phần mềm kế toán. Cần lưu ý các bước nhập liệu hóa đơn VAT.",
            "metadata": {"source": "test2.txt"}
        },
        {
            "text": "Thông tin về dự án Alpha-Bravo-Charlie. Người phụ trách: Lê Hoàng.",
            "metadata": {"source": "test3.txt"}
        }
    ]
    
    doc_id = "test_hybrid_doc_123"
    add_documents(test_docs, doc_id)
    logger.info("Test documents added.")

    # 3. Test Keyword matching (Sparse)
    # Search for a specific code that embedding might not favor but BM25 should
    query_keyword = "MAINT-2024-SERVER"
    results = search(query_keyword, top_k=1)
    
    logger.info(f"Search for keyword '{query_keyword}':")
    if results and "MAINT-2024-SERVER" in results[0]["text"]:
        logger.success("PASSED: Keyword found via BM25.")
    else:
        logger.error("FAILED: Keyword not found in top result.")

    # 4. Test Semantic matching (Dense)
    query_semantic = "làm thế nào để nhập hóa đơn?"
    results = search(query_semantic, top_k=1)
    
    logger.info(f"Search for semantic '{query_semantic}':")
    if results and "kế toán" in results[0]["text"]:
        logger.success("PASSED: Semantic match found via Vector Search.")
    else:
        logger.error("FAILED: Semantic match not found.")

    # 5. Clean up
    collection.delete(ids=[f"{doc_id}_chunk_{i}" for i in range(len(test_docs))])
    logger.info("Test cleanup finished.")

if __name__ == "__main__":
    test_hybrid_retrieval()
