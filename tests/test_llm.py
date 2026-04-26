import requests
import time

print("Testing LLM mode search...")
start = time.time()

try:
    r = requests.post(
        "http://localhost:8000/api/search",
        json={"query": "Quy trinh nghi phep nhu the nao?", "top_k": 3},
        timeout=120
    )
    elapsed = time.time() - start
    data = r.json()
    
    print(f"Status: {r.status_code}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Mode: {data.get('generation_mode')}")
    print(f"Timings: {data.get('timings_ms')}")
    print(f"Needs clarification: {data.get('needs_clarification')}")
    answer = data.get("ai_answer", "")
    print(f"Answer (first 500 chars): {answer[:500]}")
    print(f"Sources: {data.get('ai_sources')}")
except Exception as e:
    elapsed = time.time() - start
    print(f"ERROR after {elapsed:.1f}s: {e}")
