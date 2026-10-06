"""
Per-document original-vs-summary cosine similarity (Reviewer 2 #8).

The summary task's headline hallucination signal in the paper is the
semantic similarity between the source article and the generated summary
(§3.4 worked example, §4.3.1, Figure 3 CDF) -- which is precisely the
figure Reviewer 2 #8 points at when asking whether the GPT-3.5 vs
GPT-4o-mini gaps are real. The published pipeline only ever persisted the
low-similarity tail (`low_cosine_similarity*.json`, cosine <= 0.8), so the
paired bootstrap had no full per-document vector to run on.

This module regenerates the *full* per-document cosine for a
`gpt_results_*.json` file and writes it in the shape `stats.py` consumes:

  {"<doc_id>": {"_overall_": {"cosine_sim": 0.93...}}, ...}

so the comparison is a direct call into the existing stats CLI:

    python -m final_pipeline.analyzer.cosine_similarity \
        --input gpt_results_2503.json \
        --output notes/doc_cosine_gpt4omini.json

    python -m final_pipeline.analyzer.cosine_similarity \
        --input gpt_results_250427.json \
        --output notes/doc_cosine_gpt35.json

    python -m final_pipeline.analyzer.stats \
        --model-a gpt-3.5-turbo --model-b gpt-4o-mini \
        --metrics-a notes/doc_cosine_gpt35.json \
        --metrics-b notes/doc_cosine_gpt4omini.json \
        --metrics cosine_sim --labels _overall_ \
        --output notes/stats_results_summary_cosine.md

The embedding recipe mirrors `cosine_similarity.ipynb` exactly -- same
model, same CLS-token pooling, same truncation -- so the regenerated
values stay consistent with the already-published Figure 3. Deviating
from that (e.g. `--pooling mean`) changes the numbers and should only be
used for a deliberate sensitivity check.

Runs are checkpointed: a 4,940-document pass is ~10k forward passes, so
`--checkpoint-every` flushes partial results and a re-run resumes where
it left off.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_MODEL = "jinaai/jina-embeddings-v3"
METRIC_KEY = "cosine_sim"


def resolve_device(requested: str = "auto") -> str:
    """Pick a torch device, preferring CUDA, then Apple MPS, then CPU."""
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pairs(
    input_path: Path,
    original_key: str = "original_text",
    summary_key: str = "summary_text",
    id_key: str = "id",
) -> List[Tuple[str, str, str]]:
    """Return [(doc_id, original_text, summary_text)] skipping empty rows."""
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    pairs: List[Tuple[str, str, str]] = []
    for entry in data:
        original = (entry.get(original_key) or "").strip()
        summary = (entry.get(summary_key) or "").strip()
        if not original or not summary:
            # A doc with no summary has no defined original-vs-summary
            # similarity; dropping it here keeps the paired test honest
            # rather than injecting a 0.0 that would bias the mean.
            continue
        pairs.append((str(entry[id_key]), original, summary))
    return pairs


def _pool(last_hidden_state, attention_mask, pooling: str):
    """CLS-token pooling (notebook parity) or attention-masked mean pooling."""
    import torch

    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two equally shaped matrices."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    num = (a * b).sum(axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    denom = np.where(denom == 0.0, np.nan, denom)
    return num / denom


def compute_cosines(
    pairs: Sequence[Tuple[str, str, str]],
    model_name: str = DEFAULT_MODEL,
    device: str = "auto",
    batch_size: int = 8,
    max_length: int = 8192,
    pooling: str = "cls",
    fp16: Optional[bool] = None,
    checkpoint_path: Optional[Path] = None,
    checkpoint_every: int = 25,
    done: Optional[Dict[str, float]] = None,
    progress: bool = True,
) -> Dict[str, float]:
    """Embed each (original, summary) pair and return {doc_id: cosine}."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    resolved = resolve_device(device)
    # fp16 is a CUDA-era optimisation from the notebook; MPS/CPU are more
    # numerically reliable in fp32 and the speed difference is minor there.
    use_fp16 = (resolved == "cuda") if fp16 is None else fp16

    results: Dict[str, float] = dict(done or {})
    pending = [p for p in pairs if p[0] not in results]
    if not pending:
        return results

    dtype = torch.float16 if use_fp16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # The dtype is applied *after* loading rather than via a from_pretrained
    # kwarg on purpose: transformers>=5 renamed `torch_dtype` to `dtype`, and
    # passing either one leaks a torch.dtype object into jina-v3's remote
    # config code, which does `hasattr(torch, torch_dtype)` and therefore
    # requires a string. Loading at the checkpoint's own dtype and casting
    # here sidesteps the version skew entirely.
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.to(device=resolved, dtype=dtype)
    model.eval()

    iterator = range(0, len(pending), batch_size)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc=f"cosine[{resolved}]", unit="batch")
        except ImportError:
            pass

    since_flush = 0
    for start in iterator:
        batch = pending[start : start + batch_size]
        ids = [doc_id for doc_id, _, _ in batch]
        # Originals and summaries share one forward pass so both sides of a
        # pair are embedded under identical padding conditions.
        texts = [o for _, o, _ in batch] + [s for _, _, s in batch]

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(resolved)

        with torch.no_grad():
            outputs = model(**inputs)

        pooled = _pool(
            outputs.last_hidden_state,
            inputs["attention_mask"],
            pooling,
        ).float().cpu().numpy()

        half = len(batch)
        sims = _cosine_rows(pooled[:half], pooled[half:])
        for doc_id, sim in zip(ids, sims):
            if not math.isnan(float(sim)):
                results[doc_id] = float(sim)

        since_flush += 1
        if checkpoint_path and since_flush >= checkpoint_every:
            _write_checkpoint(checkpoint_path, results)
            since_flush = 0

    if checkpoint_path:
        _write_checkpoint(checkpoint_path, results)
    return results


def _write_checkpoint(path: Path, results: Dict[str, float]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(results, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_checkpoint(path: Optional[Path]) -> Dict[str, float]:
    if not path or not Path(path).exists():
        return {}
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {str(k): float(v) for k, v in raw.items()}


def to_doc_stats(cosines: Dict[str, float]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Wrap {doc_id: cosine} into the nested shape stats.py consumes."""
    return {
        str(doc_id): {"_overall_": {METRIC_KEY: float(value)}}
        for doc_id, value in cosines.items()
    }


def summarize(cosines: Dict[str, float], low_threshold: float = 0.8) -> dict:
    """Descriptive stats, including the <=threshold tail the paper reports."""
    values = np.asarray(list(cosines.values()), dtype=np.float64)
    if values.size == 0:
        return {"n": 0}
    below = int((values <= low_threshold).sum())
    return {
        "n": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
        f"n_below_{low_threshold}": below,
        f"pct_below_{low_threshold}": float(below) / float(values.size) * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Full per-document original-vs-summary cosine similarity"
    )
    parser.add_argument("--input", required=True, help="gpt_results_*.json")
    parser.add_argument("--output", required=True, help="stats.py-ready doc-stats JSON")
    parser.add_argument("--raw-output", help="Optional {doc_id: cosine} flat JSON")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--pooling", default="cls", choices=["cls", "mean"],
        help="cls matches cosine_similarity.ipynb / published Figure 3",
    )
    parser.add_argument("--fp16", dest="fp16", action="store_true", default=None)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--id-key", default="id")
    parser.add_argument("--original-key", default="original_text")
    parser.add_argument("--summary-key", default="summary_text")
    parser.add_argument(
        "--checkpoint", help="Partial-result path; re-running resumes from it"
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--limit", type=int, help="Only process the first N pairs")
    args = parser.parse_args()

    pairs = load_pairs(
        Path(args.input),
        original_key=args.original_key,
        summary_key=args.summary_key,
        id_key=args.id_key,
    )
    if args.limit:
        pairs = pairs[: args.limit]

    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    done = _read_checkpoint(checkpoint)
    if done:
        print(f"[resume] {len(done)} documents already computed")

    print(f"[input] {len(pairs)} (original, summary) pairs from {args.input}")

    cosines = compute_cosines(
        pairs,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        pooling=args.pooling,
        fp16=args.fp16,
        checkpoint_path=checkpoint,
        checkpoint_every=args.checkpoint_every,
        done=done,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(to_doc_stats(cosines), indent=2), encoding="utf-8"
    )

    if args.raw_output:
        raw_path = Path(args.raw_output)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(cosines, indent=2), encoding="utf-8")

    stats = summarize(cosines)
    print(f"[done] wrote {out_path}")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
