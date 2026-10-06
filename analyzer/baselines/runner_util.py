"""
Shared execution helpers for the FactScore / QAGS baseline runners (R2 #6).

A full 200-doc baseline run is ~4.4K judge calls. Sequentially that is
hours, and a single transient API error would throw away everything
because the runners only write their output at the very end. This module
adds the three things such a run needs:

* ``retry`` — exponential backoff around one judge call
* ``load_done`` / ``append_done`` — per-doc JSONL checkpoint so a killed
  run resumes where it stopped
* ``run_docs`` — bounded thread pool over docs (each doc's own calls stay
  sequential, so per-doc results are unchanged by the parallelism)
"""

from __future__ import annotations

import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

MAX_ATTEMPTS = 5
BASE_DELAY = 2.0


def retry(fn: Callable[[], str], what: str = "judge call") -> str:
    """Call ``fn`` with exponential backoff + jitter on any exception."""
    last: Optional[Exception] = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — retry on anything the SDK raises
            last = e
            if attempt == MAX_ATTEMPTS:
                break
            delay = BASE_DELAY * (2 ** (attempt - 1)) + random.uniform(0, 1)
            print(f"[retry] {what} failed ({attempt}/{MAX_ATTEMPTS}): {e} "
                  f"— sleeping {delay:.1f}s", file=sys.stderr, flush=True)
            time.sleep(delay)
    raise RuntimeError(f"{what} failed after {MAX_ATTEMPTS} attempts: {last}")


def ckpt_path(output_path: str) -> Path:
    return Path(str(output_path) + ".partial.jsonl")


def load_done(output_path: str) -> Dict[str, dict]:
    """Read already-scored docs from the checkpoint (doc_id -> record)."""
    p = ckpt_path(output_path)
    if not p.exists():
        return {}
    done: Dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated final line from a hard kill
        if rec.get("doc_id"):
            done[str(rec["doc_id"])] = rec
    return done


def run_docs(docs: List[dict], output_path: str, score_fn: Callable[[dict], dict],
             workers: int = 4, label: str = "run") -> List[dict]:
    """
    Score every doc, skipping ones already in the checkpoint.

    Results come back in the original ``docs`` order regardless of the
    order the threads finish in.
    """
    done = load_done(output_path)
    todo = [d for d in docs if str(d.get("id")) not in done]
    if done:
        print(f"[{label}] resuming — {len(done)} cached, {len(todo)} to go",
              file=sys.stderr, flush=True)

    lock = threading.Lock()
    ckpt = ckpt_path(output_path)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    n_done = 0

    def _work(doc: dict) -> dict:
        nonlocal n_done
        try:
            rec = score_fn(doc)
        except Exception as e:  # noqa: BLE001 — never lose the rest of the run
            rec = {"doc_id": str(doc.get("id")), "note": f"failed: {e}"}
        with lock:
            with open(ckpt, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
            n_done += 1
            if n_done % 5 == 0 or n_done == len(todo):
                print(f"[{label}] {n_done}/{len(todo)}", file=sys.stderr, flush=True)
        return rec

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_work, d) for d in todo]
            for fut in as_completed(futures):
                rec = fut.result()
                done[str(rec.get("doc_id"))] = rec

    ordered = [done[str(d.get("id"))] for d in docs if str(d.get("id")) in done]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(ordered, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    return ordered


def load_subset(subset_path: str, limit: Optional[int] = None) -> List[dict]:
    payload = json.loads(Path(subset_path).read_text(encoding="utf-8"))
    docs = payload.get("docs", [])
    return docs[:limit] if limit is not None else docs
