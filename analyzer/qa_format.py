"""Normalise QA answers into the pipeline's canonical 5-field form.

The QA prompt asks for ``{Answer1, Answer2, Answer3, Answer4, Answer5}``.
GPT-3.5-turbo and GPT-4o-mini comply on ~99% of documents, so
``extract_doc_stats`` could simply split on commas. Llama-3-8B-Instruct
complies on ~0%: it answers correctly but wraps the answers in prose
("Here are the answers to the questions:", "Answer 1: Ducktail", or a full
echo of each question followed by its answer).

Scoring that as 100% format failure would report an instruction-following
gap as a hallucination measurement, so this module converts the common
layouts into the canonical string. It is a **no-op for already-canonical
output**, which keeps the GPT numbers bit-identical to the published ones.

The format-compliance rate itself is a finding worth reporting, so
``compliance_report`` returns the raw counts alongside the normalised text.

Commas inside an answer would break the downstream 5-way split, so they are
replaced with spaces (the entity-matching metrics tokenise anyway).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

N_ANSWERS = 5

_CANONICAL = re.compile(r"^\s*\{[^{}]*\}\s*$", re.S)
# "Answer 1: foo" / "Answer 1 : foo" / "**Answer 1:** foo"
_ANSWER_LINE = re.compile(
    r"(?:\*\*)?answer\s*(\d)\s*(?:\*\*)?\s*[::]\s*(?:\*\*)?\s*(.+?)\s*(?:\*\*)?$",
    re.I | re.M,
)
# "Question 3 : What is ...?" — the answer is whatever follows on the next lines
_QUESTION_LINE = re.compile(r"(?:\*\*)?question\s*(\d)\s*(?:\*\*)?\s*[::]", re.I)
# The question text itself, echoed without a "Question N:" prefix.
_QUESTION_KEYWORD = re.compile(
    r"main\s+(MALWARE|SYSTEM|VULNERABILITY|INDICATOR|ORGANIZATION)\b", re.I
)
_KEYWORD_INDEX = {  # prompt order, not LABELS order
    "MALWARE": 1, "SYSTEM": 2, "VULNERABILITY": 3, "INDICATOR": 4, "ORGANIZATION": 5,
}


def _clean_answer(text: str) -> str:
    text = text.strip().strip("`").strip()
    text = re.sub(r"^\*+|\*+$", "", text).strip()
    text = text.replace(",", " ")           # keep the 5-way split intact
    text = re.sub(r"\s+", " ", text)
    if text.lower() in {"none", "n/a", "not applicable", "no answer.", ""}:
        return "no answer"
    return text


def parse_answers(qa_text: str) -> Optional[List[str]]:
    """Return the five answers in prompt order, or None if unrecoverable."""
    if not qa_text or not qa_text.strip():
        return None

    text = qa_text.strip()

    # 1. Canonical "{a, b, c, d, e}"
    if _CANONICAL.match(text):
        parts = [p.strip() for p in text.strip("{} \n").split(",")]
        if len(parts) == N_ANSWERS:
            return parts
        # A comma inside an answer produced too many fields — unrecoverable
        # without guessing, so fall through to the prose parsers.

    # 2. "Answer N: ..." lines
    found: Dict[int, str] = {}
    for m in _ANSWER_LINE.finditer(text):
        idx = int(m.group(1))
        if 1 <= idx <= N_ANSWERS and idx not in found:
            found[idx] = _clean_answer(m.group(2))
    if len(found) == N_ANSWERS:
        return [found[i] for i in range(1, N_ANSWERS + 1)]

    # 3. "Question N: <question>\n<answer>" — take the first non-empty line
    #    after each question that is not itself a question.
    lines = text.splitlines()
    q_positions = []
    for i, line in enumerate(lines):
        m = _QUESTION_LINE.search(line)
        if m:
            q_positions.append((int(m.group(1)), i))
    if q_positions:
        for idx, pos in q_positions:
            if not (1 <= idx <= N_ANSWERS) or idx in found:
                continue
            # answer may sit on the same line after the '?' or on later lines
            same_line = lines[pos].split("?", 1)
            candidate = same_line[1] if len(same_line) > 1 and same_line[1].strip() else ""
            if not candidate:
                for nxt in lines[pos + 1:]:
                    if not nxt.strip():
                        continue
                    if _QUESTION_LINE.search(nxt):
                        break
                    candidate = nxt
                    break
            if candidate:
                found[idx] = _clean_answer(candidate)
        if len(found) == N_ANSWERS:
            return [found[i] for i in range(1, N_ANSWERS + 1)]

    # 4. The question text echoed without a "Question N:" prefix
    #    ("What is the main VULNERABILITY covered in this article?\nNo answer").
    #    Prompt order is MALWARE, SYSTEM, VULNERABILITY, INDICATOR, ORGANIZATION.
    kw_hits = []
    for i, line in enumerate(lines):
        m = _QUESTION_KEYWORD.search(line)
        if m:
            kw_hits.append((_KEYWORD_INDEX[m.group(1).upper()], i))
    if kw_hits:
        for idx, pos in kw_hits:
            if idx in found:
                continue
            tail = lines[pos].split("?", 1)
            candidate = tail[1] if len(tail) > 1 and tail[1].strip() else ""
            if not candidate:
                for nxt in lines[pos + 1:]:
                    if not nxt.strip():
                        continue
                    if _QUESTION_KEYWORD.search(nxt):
                        break
                    candidate = nxt
                    break
            if candidate:
                found[idx] = _clean_answer(candidate)
        if len(found) == N_ANSWERS:
            return [found[i] for i in range(1, N_ANSWERS + 1)]

    return None


def normalize_qa_text(qa_text: str) -> str:
    """Canonical ``{a1, a2, a3, a4, a5}`` string, or '' when unrecoverable."""
    answers = parse_answers(qa_text)
    if answers is None:
        return ""
    return "{" + ", ".join(answers) + "}"


def compliance_report(records: List[dict], qa_key: str = "qa_text") -> Dict[str, int]:
    """Counts of already-canonical / recovered / unrecoverable QA outputs."""
    canonical = recovered = failed = 0
    for r in records:
        text = r.get(qa_key) or ""
        is_canonical = bool(_CANONICAL.match(text.strip())) and len(
            [p for p in text.strip("{} \n").split(",") if p.strip()]
        ) == N_ANSWERS
        if is_canonical:
            canonical += 1
        elif parse_answers(text) is not None:
            recovered += 1
        else:
            failed += 1
    return {
        "n": len(records),
        "canonical": canonical,
        "recovered": recovered,
        "unrecoverable": failed,
    }
