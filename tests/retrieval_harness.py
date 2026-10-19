"""
Retrieval Benchmark Harness (offline, không cần LLM/server).

Đo chất lượng retrieval hai tầng đúng như production:
  1. Hybrid search (dense + BM25 + legal boost)  -> hybrid@K
  2. Rerank (CrossEncoder) top-N                 -> R@1, R@3, R@N, MRR@N

Gold định nghĩa theo "must" (các chuỗi phải cùng xuất hiện trong 1 chunk)
+ file khớp tiền tố — bền vững qua re-ingest (không phụ thuộc chunk_id).

Chạy:
    python tests/retrieval_harness.py
    BENCH_SEARCH_K=18 BENCH_RERANK_N=6 python tests/retrieval_harness.py
    BENCH_SKIP_RERANK=1 python tests/retrieval_harness.py   # chỉ đo hybrid
Kết quả JSON: logs/retrieval_bench_last.json (đổi bằng BENCH_OUT).
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(override=False)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retrieval_bench.json")
SEARCH_K = int(os.getenv("BENCH_SEARCH_K", "18"))
RERANK_N = int(os.getenv("BENCH_RERANK_N", "6"))
SKIP_RERANK = os.getenv("BENCH_SKIP_RERANK", "0") == "1"
BENCH_OUT = os.getenv("BENCH_OUT", os.path.join(BASE_DIR, "logs", "retrieval_bench_last.json"))


def _is_hit(result: dict, gold_files: list, must: list) -> bool:
    meta = result.get("metadata") or {}
    file_name = (result.get("file") or meta.get("source") or "").lower()
    if not any(g.lower() in file_name for g in gold_files):
        return False
    text = (result.get("text") or "").casefold()
    return all(m.casefold() in text for m in must)


def _first_hit_rank(results: list, gold_files: list, must: list) -> int:
    for idx, r in enumerate(results, start=1):
        if _is_hit(r, gold_files, must):
            return idx
    return 0


def main():
    with open(BENCH_FILE, encoding="utf-8") as f:
        bench = json.load(f)

    from backend.vector_store import search
    from backend.generator import rerank_results

    print("=" * 72)
    print(f" RETRIEVAL HARNESS — {len(bench)} queries | search_k={SEARCH_K} rerank_n={RERANK_N} skip_rerank={SKIP_RERANK}")
    print("=" * 72)

    rows = []
    hybrid_hits = 0
    r1 = r3 = rn = 0
    mrr_sum = 0.0

    for item in bench:
        qid, query = item["id"], item["query"]
        gold_files, must = item["gold_files"], item.get("must", [])

        t0 = time.perf_counter()
        results = search(query, top_k=SEARCH_K)
        t_search = time.perf_counter() - t0

        h_rank = _first_hit_rank(results, gold_files, must)
        hybrid_hits += 1 if h_rank else 0

        row = {
            "id": qid,
            "hybrid_hit": bool(h_rank),
            "hybrid_rank": h_rank,
            "search_s": round(t_search, 2),
        }

        if not SKIP_RERANK:
            t0 = time.perf_counter()
            reranked = rerank_results(query, results, top_n=RERANK_N)
            t_rerank = time.perf_counter() - t0

            r_rank = _first_hit_rank(reranked, gold_files, must)
            row.update({
                "rerank_rank": r_rank,
                "rerank_s": round(t_rerank, 2),
                "rerank_hit": bool(r_rank),
            })
            if r_rank:
                r1 += 1 if r_rank == 1 else 0
                r3 += 1 if r_rank <= 3 else 0
                rn += 1
                mrr_sum += 1.0 / r_rank

        rows.append(row)
        status = "HIT " if h_rank else "MISS"
        extra = f" rank={h_rank}" if h_rank else ""
        rr = f" | rerank={'HIT rank=' + str(row['rerank_rank']) if row.get('rerank_hit') else 'MISS'}" if not SKIP_RERANK else ""
        print(f"  [{status}] {qid:<22}{extra}{rr}  ({row['search_s']}s)")

    n = len(bench)
    summary = {
        "n_queries": n,
        "search_k": SEARCH_K,
        "rerank_n": RERANK_N,
        "hybrid_at_k": round(hybrid_hits / n, 3),
    }
    if not SKIP_RERANK:
        summary.update({
            "recall_at_1": round(r1 / n, 3),
            "recall_at_3": round(r3 / n, 3),
            "recall_at_n": round(rn / n, 3),
            "mrr_at_n": round(mrr_sum / n, 3),
        })

    print("-" * 72)
    print(f"  hybrid@{SEARCH_K}: {summary['hybrid_at_k']:.3f}", end="")
    if not SKIP_RERANK:
        print(f" | R@1: {summary['recall_at_1']:.3f} | R@3: {summary['recall_at_3']:.3f} "
              f"| R@{RERANK_N}: {summary['recall_at_n']:.3f} | MRR@{RERANK_N}: {summary['mrr_at_n']:.3f}")
    else:
        print()
    print("=" * 72)

    report = {"summary": summary, "rows": rows}
    os.makedirs(os.path.dirname(BENCH_OUT), exist_ok=True)
    with open(BENCH_OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report: {BENCH_OUT}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
