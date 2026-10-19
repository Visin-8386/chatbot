"""
Batch re-ingest toàn bộ corpus thật bằng chunker hiện tại (legal-aware + fix split).

Với mỗi tài liệu đang có trong Chroma: xóa chunks cũ -> process_document lại ->
add_documents với doc_id cũ (rebuild_bm25=False), cuối cùng rebuild BM25 1 lần.

Chạy nền (khuyến nghị):
    python -u tests/reingest_corpus.py
Biến môi trường:
    REINGEST_SKIP_DOC_IDS  - doc_id bỏ qua, phân tách bởi "," (mặc định "8a52a6d0" = ND293 đã re-ingest)
    REINGEST_CORPUS_DIR    - thư mục corpus (mặc định test_data/real_corpus)
    REINGEST_SUMMARY       - file JSON tóm tắt kết quả (mặc định logs/reingest_summary.json)
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(override=False)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.getenv("REINGEST_CORPUS_DIR", os.path.join(BASE_DIR, "test_data", "real_corpus"))
SKIP_DOC_IDS = {d.strip() for d in os.getenv("REINGEST_SKIP_DOC_IDS", "8a52a6d0").split(",") if d.strip()}
SUMMARY_FILE = os.getenv("REINGEST_SUMMARY", os.path.join(BASE_DIR, "logs", "reingest_summary.json"))


def main() -> int:
    from backend.vector_store import (
        get_collection, add_documents, delete_document, _initialize_bm25,
    )
    from backend.document_processor import process_document

    collection = get_collection()
    data = collection.get(include=["metadatas"])

    doc_sources = {}
    for meta in data["metadatas"]:
        did = meta.get("doc_id")
        if did and did not in doc_sources:
            doc_sources[did] = meta.get("source", "")

    corpus_files = {
        name.lower(): os.path.join(CORPUS_DIR, name)
        for name in os.listdir(CORPUS_DIR)
        if os.path.isfile(os.path.join(CORPUS_DIR, name))
    }

    todo, missing, skipped = [], [], []
    for did, src in sorted(doc_sources.items()):
        if did in SKIP_DOC_IDS:
            skipped.append({"doc_id": did, "source": src, "reason": "skip-list"})
            continue
        if not src or src.startswith("_"):
            skipped.append({"doc_id": did, "source": src, "reason": "manifest/non-doc"})
            continue
        path = corpus_files.get(src.lower())
        if not path:
            missing.append({"doc_id": did, "source": src})
            continue
        todo.append((did, src, path))

    print(f"docs in store: {len(doc_sources)} | todo: {len(todo)} | "
          f"missing file: {len(missing)} | skipped: {len(skipped)}", flush=True)
    for m in missing:
        print(f"  MISSING FILE: {m['doc_id']} | {m['source']}", flush=True)

    started = time.time()
    results = []
    ok = fail = 0

    for idx, (doc_id, source, path) in enumerate(todo, start=1):
        t0 = time.time()
        try:
            n_old = delete_document(doc_id, rebuild_bm25=False)
            chunks = process_document(path)
            if not chunks:
                raise ValueError("no text content extracted")
            n_new = add_documents(chunks, doc_id, rebuild_bm25=False)
            ok += 1
            results.append({
                "doc_id": doc_id, "source": source,
                "old_chunks": n_old, "new_chunks": n_new,
                "seconds": round(time.time() - t0, 1),
            })
            print(f"[{idx}/{len(todo)}] OK  {doc_id} | {n_old:>5} -> {n_new:>5} | "
                  f"{time.time() - t0:5.1f}s | elapsed {time.time() - started:6.0f}s | {source[:70]}",
                  flush=True)
        except Exception as e:
            fail += 1
            results.append({
                "doc_id": doc_id, "source": source, "error": repr(e),
                "seconds": round(time.time() - t0, 1),
            })
            print(f"[{idx}/{len(todo)}] FAIL {doc_id} | {e!r} | {source[:70]}", flush=True)

    print("Rebuilding BM25 index once (force)...", flush=True)
    _initialize_bm25(force_rebuild=True)
    total_chunks = collection.count()
    print(f"DONE ok={ok} fail={fail} elapsed={time.time() - started:.0f}s | "
          f"total chunks now: {total_chunks}", flush=True)

    summary = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
        "elapsed_seconds": round(time.time() - started, 1),
        "ok": ok, "fail": fail,
        "total_chunks_after": total_chunks,
        "missing_files": missing,
        "skipped": skipped,
        "results": results,
    }
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Summary: {SUMMARY_FILE}", flush=True)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
