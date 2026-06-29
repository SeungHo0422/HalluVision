"""
Span-level matching utilities (Reviewer 2 #10).

The original QA evaluation forces a single-entity exact lowercase match,
which collapses precision on Indicator / Vulnerability labels — those
naturally carry multi-token, format-variant answers (CVE-2024-1234,
sha256 hashes, IP addresses). This module adds three relaxed match modes
that separate NER-side surface variance from LLM-side hallucination.

Modes:
- ``exact`` — legacy behaviour: ``pred.lower() in {g.lower() for g in gold}``
- ``token_f1`` — token-set F1 between the predicted span and each gold
  entity; the best F1 across gold entities is returned
- ``jaccard`` — token-set Jaccard between predicted span and each gold
- ``relaxed`` — substring containment (either direction) OR token-F1 >=
  threshold; useful when the LLM emits "CVE-2024-1234, CVE-2024-5678"
  for a single-answer slot

CLI / helper ``score_doc_label`` lets the downstream analyzer call this
uniformly via ``match_mode`` argument.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Sequence, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[.\-_][A-Za-z0-9]+)*")


def tokenize(span: str) -> List[str]:
    """Lowercased alphanumeric tokens, preserving CVE/hash dot-dash forms."""
    if not span:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(span)]


def token_f1(pred: str, gold: str) -> float:
    p_tokens = tokenize(pred)
    g_tokens = tokenize(gold)
    if not p_tokens or not g_tokens:
        return 0.0
    p_set, g_set = set(p_tokens), set(g_tokens)
    common = p_set & g_set
    if not common:
        return 0.0
    p = len(common) / len(p_set)
    r = len(common) / len(g_set)
    return 2 * p * r / (p + r)


def jaccard(pred: str, gold: str) -> float:
    p_set = set(tokenize(pred))
    g_set = set(tokenize(gold))
    if not p_set and not g_set:
        return 0.0
    union = p_set | g_set
    if not union:
        return 0.0
    return len(p_set & g_set) / len(union)


def best_token_f1(pred: str, gold_entities: Iterable[str]) -> float:
    return max((token_f1(pred, g) for g in gold_entities), default=0.0)


def best_jaccard(pred: str, gold_entities: Iterable[str]) -> float:
    return max((jaccard(pred, g) for g in gold_entities), default=0.0)


def relaxed_match(
    pred: str,
    gold_entities: Iterable[str],
    f1_threshold: float = 0.5,
) -> bool:
    """Match if substring containment in either direction OR token-F1 >= threshold."""
    p_low = (pred or "").lower().strip()
    if not p_low:
        return False
    for g in gold_entities:
        g_low = (g or "").lower().strip()
        if not g_low:
            continue
        if p_low == g_low or p_low in g_low or g_low in p_low:
            return True
        if token_f1(pred, g) >= f1_threshold:
            return True
    return False


def exact_match(pred: str, gold_entities: Iterable[str]) -> bool:
    p_low = (pred or "").lower().strip()
    if not p_low:
        return False
    return p_low in {(g or "").lower().strip() for g in gold_entities}


def score_doc_label(
    pred: str,
    gold_entities: Sequence[str],
    match_mode: str = "exact",
    f1_threshold: float = 0.5,
) -> Tuple[int, float]:
    """
    Return (binary_match, soft_score). ``binary_match`` is the
    boolean-as-int used by the legacy P/R counting; ``soft_score`` is
    the underlying continuous score (token-F1 / Jaccard) when applicable
    — 1.0 for binary modes.

    Convention: "no answer" → (0, 0.0).
    """
    if not pred or pred.lower().strip() == "no answer":
        return 0, 0.0
    if match_mode == "exact":
        return (1 if exact_match(pred, gold_entities) else 0), 1.0 if exact_match(pred, gold_entities) else 0.0
    if match_mode == "token_f1":
        s = best_token_f1(pred, gold_entities)
        return (1 if s >= f1_threshold else 0), s
    if match_mode == "jaccard":
        s = best_jaccard(pred, gold_entities)
        return (1 if s >= f1_threshold else 0), s
    if match_mode == "relaxed":
        return (1 if relaxed_match(pred, gold_entities, f1_threshold) else 0), 1.0 if relaxed_match(pred, gold_entities, f1_threshold) else 0.0
    raise ValueError(f"Unknown match_mode: {match_mode}")
