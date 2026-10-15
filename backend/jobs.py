"""
Jobs - Background ingest jobs chạy trong process (stdlib-only, không cần Redis).

Vì sao không Celery ngay: single-machine + 6GB VRAM + offline -> thêm Redis/worker
riêng tốn RAM và phức tạp deploy. Job chạy trên ThreadPool 1 worker để Docling
không bao giờ xử lý 2 file nặng cùng lúc. Khi cần scale multi-replica, thay
JobManager bằng Celery mà không đổi API (/api/upload, /api/jobs/{id}).

Mặc định TẮT (ENABLE_ASYNC_INGEST=0): /api/upload xử lý đồng bộ như cũ.
"""
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional
from loguru import logger


class JobManager:
    def __init__(self, max_workers: int = 1):
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, max_workers), thread_name_prefix="ingest"
        )
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def submit(self, label: str, fn: Callable[[], Dict[str, Any]]) -> str:
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "label": label,
                "status": "queued",
                "created_at": time.time(),
                "finished_at": None,
                "result": None,
                "error": None,
            }
        self._executor.submit(self._run, job_id, fn)
        return job_id

    def _run(self, job_id: str, fn: Callable[[], Dict[str, Any]]) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job["status"] = "running"
        try:
            result = fn()
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["status"] = "done"
                    job["finished_at"] = time.time()
                    job["result"] = result
        except Exception as e:
            logger.error("Job {} failed: {}", job_id, e)
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["status"] = "error"
                    job["finished_at"] = time.time()
                    job["error"] = f"{type(e).__name__}: {e}"

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None

    def list(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            jobs = sorted(
                self._jobs.values(), key=lambda j: j["created_at"], reverse=True
            )
        return [dict(j) for j in jobs[: max(1, limit)]]

    def prune(self, keep: int = 200) -> int:
        """Giữ tối đa `keep` job mới nhất để không phình RAM. Trả số đã xóa."""
        with self._lock:
            if len(self._jobs) <= keep:
                return 0
            ordered = sorted(self._jobs.items(), key=lambda kv: kv[1]["created_at"])
            for job_id, _job in ordered[: len(self._jobs) - keep]:
                del self._jobs[job_id]
            return len(ordered) - keep


job_manager = JobManager(max_workers=1)


def traceback_str() -> str:
    return traceback.format_exc(limit=3)
