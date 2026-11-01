"""Background job execution with progress reporting.

Heavy operations run on a single worker thread (bounded memory use); the UI
polls job status. Errors are captured with their user-facing message.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from ..errors import LumenError

log = logging.getLogger("lumen.jobs")
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lumen-job")
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def submit(kind: str, fn: Callable[[Callable[[float, str], None]], Any]) -> str:
    job_id = uuid.uuid4().hex[:12]
    job = {"id": job_id, "kind": kind, "state": "queued", "progress": 0.0, "message": "Waiting…", "result": None, "error": None, "created": time.time()}
    with _lock:
        _jobs[job_id] = job
        # forget old finished jobs
        for k in [k for k, j in _jobs.items() if j["state"] in ("done", "error") and time.time() - j["created"] > 3600]:
            _jobs.pop(k, None)

    def progress(p: float, msg: str = "") -> None:
        job["progress"] = float(max(job["progress"], min(1.0, p)))
        if msg:
            job["message"] = msg

    def run():
        job["state"] = "running"
        job["started"] = time.time()
        try:
            job["result"] = fn(progress)
            job["state"] = "done"
            job["progress"] = 1.0
            job["message"] = "Done"
        except LumenError as exc:
            job["state"] = "error"
            job["error"] = exc.to_dict()
        except MemoryError:
            job["state"] = "error"
            job["error"] = {"code": "out_of_memory", "message": "The computer ran out of memory.", "hint": "Lower the processing resolution in Settings or close other applications."}
        except Exception as exc:  # pragma: no cover - unexpected failures
            log.error("job %s failed: %s", job_id, traceback.format_exc())
            job["state"] = "error"
            job["error"] = {"code": "internal", "message": f"Processing failed: {exc}", "hint": "This is a bug. Details were written to the log."}
        job["elapsed"] = round(time.time() - job.get("started", time.time()), 2)

    _executor.submit(run)
    return job_id


def get(job_id: str) -> dict | None:
    return _jobs.get(job_id)
