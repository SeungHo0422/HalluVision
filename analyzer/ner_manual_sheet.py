"""
Single-annotator NER gold validation sheets (Reviewer 2 #2a).

Draws a √N-sized document sample (70 of 4,940) and emits two spreadsheets a
single reviewer can work through by hand, plus one plain-text file per article:

    A_docs_recall.csv       one row per sampled doc — read the article, list the
                            entities the NER *missed* per label (recall side)
    B_entities_precision.csv one row per predicted gold entity — judge it
                            correct / partial / incorrect (precision side)
    articles/doc_<id>.txt   full article text, for comfortable reading
    README.md               labeling guide

Stratification crosses the gpt-4o-mini **summary precision** quartile with the
**QA precision** quartile (16 cells), so the sample spans the range of measured
performance rather than concentrating on easy documents. Documents are emitted
in round-robin cell order: any prefix of the sheet (the first 20 docs, say) is
still a balanced sample, so partial completion stays usable.

Each predicted entity is pre-annotated with cheap, objective evidence:

    in_text        the surface form occurs somewhere in the article
    span_flag      ok / substring_only / absent — ``substring_only`` means the
                   string appears only inside a longer token (``165`` inside
                   ``UNC2165``), the signature of a truncated CyNER span
    n_occurrences  literal occurrence count
    context        ±110 chars around the first hit

so the reviewer spends judgment on the genuinely ambiguous cells.

Fill in ``verdict`` (and optionally ``verdict_b`` for a second rater), then run
``analyzer.ner_manual_score`` to get per-label precision/recall and the
corrected F1.

Usage::

    python -m final_pipeline.analyzer.ner_manual_sheet \
        --original original_ner.json \
        --gpt-results gpt_results_2503.json \
        --summary-stats notes/doc_stats_summary_gpt4omini.json \
        --qa-stats notes/doc_stats_qa_gpt4omini.json \
        --n 70 --seed 42 \
        --outdir notes/csv/manual_ner_70
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path
from statistics import quantiles
from typing import Dict, List, Optional, Tuple

LABELS = ["Malware", "System", "Indicator", "Vulnerability", "Organization"]

CONTEXT_PAD = 110


# --------------------------------------------------------------------------- #
# stratification
# --------------------------------------------------------------------------- #

def _quartile_bucket(value: float, cutoffs: List[float]) -> int:
    for i, c in enumerate(cutoffs):
        if value <= c:
            return i
    return len(cutoffs)


def _compute_cutoffs(values: List[float]) -> List[float]:
    """Quartile cutoffs, with degenerate ones dropped.

    Summary/QA precision pile up at 1.0, so a raw quartile cutoff can equal the
    maximum and leave the top bucket permanently empty. Dropping such cutoffs
    yields fewer but genuinely populated buckets.
    """
    finite = [v for v in values if v is not None]
    if len(finite) < 4:
        return [0.25, 0.5, 0.75]
    cuts = quantiles(finite, n=4)[:3]
    top = max(finite)
    out: List[float] = []
    for c in cuts:
        if c >= top:
            continue
        if out and c <= out[-1]:
            continue
        out.append(c)
    return out


def stratified_sample(
    docs: List[dict],
    original_ner: Dict[str, dict],
    summary_stats: Dict[str, dict],
    qa_stats: Dict[str, dict],
    n: int,
    seed: int,
) -> List[dict]:
    """Sample ``n`` docs over the 4×4 (summary precision × QA precision) grid.

    Returned in round-robin cell order so any prefix stays balanced.
    """
    rng = random.Random(seed)

    eligible: List[Tuple[dict, float, float]] = []
    for d in docs:
        doc_id = str(d.get("id"))
        if doc_id not in original_ner:
            continue
        if not (d.get("original_text") or "").strip():
            continue
        s = summary_stats.get(doc_id, {}).get("_overall_", {})
        q = qa_stats.get(doc_id, {}).get("_overall_", {})
        if not s or not q:
            continue
        # a doc with no predicted entity at all gives the precision sheet
        # nothing to judge; the recall sheet alone is not worth a slot
        if not any(original_ner[doc_id].get(l) for l in LABELS):
            continue
        eligible.append((d, s.get("precision", 0.0), q.get("precision", 0.0)))

    sm_cuts = _compute_cutoffs([p for _, p, _ in eligible])
    qa_cuts = _compute_cutoffs([p for _, _, p in eligible])

    n_sm, n_qa = len(sm_cuts) + 1, len(qa_cuts) + 1
    grid: Dict[Tuple[int, int], List[dict]] = {
        (i, j): [] for i in range(n_sm) for j in range(n_qa)
    }
    for d, sm_p, qa_p in eligible:
        bucket = (_quartile_bucket(sm_p, sm_cuts), _quartile_bucket(qa_p, qa_cuts))
        enriched = dict(d)
        enriched["_sm_p_"] = sm_p
        enriched["_qa_p_"] = qa_p
        enriched["_sm_p_bucket_"] = bucket[0]
        enriched["_qa_p_bucket_"] = bucket[1]
        grid[bucket].append(enriched)
    for cell in grid.values():
        rng.shuffle(cell)

    cells = sorted(k for k, v in grid.items() if v)
    rng.shuffle(cells)

    # per-cell quota, then round-robin so a prefix of the output is balanced
    base, remainder = divmod(n, len(cells))
    picked: Dict[Tuple[int, int], List[dict]] = {}
    for idx, key in enumerate(cells):
        take = base + (1 if idx < remainder else 0)
        picked[key] = grid[key][:take]

    sampled: List[dict] = []
    round_idx = 0
    while len(sampled) < n and any(len(v) > round_idx for v in picked.values()):
        for key in cells:
            if len(picked[key]) > round_idx and len(sampled) < n:
                sampled.append(picked[key][round_idx])
        round_idx += 1

    # thin cells (rare at n=70) leave a deficit — backfill from the fullest cells
    if len(sampled) < n:
        chosen_ids = {str(d["id"]) for d in sampled}
        leftovers = [d for key in cells for d in grid[key] if str(d["id"]) not in chosen_ids]
        rng.shuffle(leftovers)
        sampled.extend(leftovers[: n - len(sampled)])

    return sampled[:n]


# --------------------------------------------------------------------------- #
# entity evidence
# --------------------------------------------------------------------------- #

def _dedup_entities(raw: List[str]) -> List[Tuple[str, int]]:
    """Case-insensitive dedup preserving first surface form; returns (form, count)."""
    seen: Dict[str, List] = {}
    order: List[str] = []
    for e in raw or []:
        form = (e or "").strip()
        if not form:
            continue
        key = form.lower()
        if key not in seen:
            seen[key] = [form, 0]
            order.append(key)
        seen[key][1] += 1
    return [(seen[k][0], seen[k][1]) for k in order]


def _boundary_match(text: str, entity: str) -> Optional[re.Match]:
    pattern = r"(?<![0-9A-Za-z])" + re.escape(entity) + r"(?![0-9A-Za-z])"
    return re.search(pattern, text, re.IGNORECASE)


def entity_evidence(text: str, entity: str) -> dict:
    """Objective, pre-computed evidence for one predicted entity."""
    lowered = text.lower()
    n_occ = lowered.count(entity.lower())
    match = _boundary_match(text, entity)

    if match is not None:
        span_flag = "ok"
        start = match.start()
    elif n_occ > 0:
        span_flag = "substring_only"
        start = lowered.find(entity.lower())
    else:
        span_flag = "absent"
        start = -1

    if start >= 0:
        left = max(0, start - CONTEXT_PAD)
        right = min(len(text), start + len(entity) + CONTEXT_PAD)
        context = re.sub(r"\s+", " ", text[left:right]).strip()
        if left > 0:
            context = "…" + context
        if right < len(text):
            context = context + "…"
    else:
        context = ""

    return {
        "in_text": "yes" if n_occ > 0 else "no",
        "span_flag": span_flag,
        "n_occurrences": n_occ,
        "context": context,
    }


# --------------------------------------------------------------------------- #
# sheet writers
# --------------------------------------------------------------------------- #

DOC_HEADER = (
    ["review_order", "doc_id", "sm_p_bucket", "qa_p_bucket",
     "summary_precision", "qa_precision", "n_gold_entities", "article_chars"]
    + [f"gold_{l}" for l in LABELS]
    + [f"missed_{l}" for l in LABELS]
    + ["doc_done", "doc_notes", "article_text", "article_truncated"]
)

ENTITY_HEADER = [
    "row_id", "review_order", "doc_id", "label", "entity",
    "n_in_ner_list", "also_labeled_as", "in_text", "span_flag", "n_occurrences",
    "weight", "context", "verdict", "error_type", "correct_form", "notes", "verdict_b",
]


def write_sheets(
    sampled: List[dict],
    original_ner: Dict[str, dict],
    outdir: Path,
    max_article_chars: int,
    write_articles: bool,
    max_entities_per_doc: int = 0,
    seed: int = 42,
    always_labels: Optional[List[str]] = None,
) -> Tuple[int, int]:
    outdir.mkdir(parents=True, exist_ok=True)
    art_dir = outdir / "articles"
    if write_articles:
        art_dir.mkdir(parents=True, exist_ok=True)

    doc_rows: List[List] = []
    ent_rows: List[List] = []
    row_id = 0
    rng = random.Random(seed)

    for order, d in enumerate(sampled, start=1):
        doc_id = str(d["id"])
        text = d.get("original_text") or ""
        ner = original_ner.get(doc_id, {})

        per_label = {l: _dedup_entities(ner.get(l) or []) for l in LABELS}
        n_gold = sum(len(v) for v in per_label.values())

        if write_articles:
            (art_dir / f"doc_{doc_id}.txt").write_text(text, encoding="utf-8")

        # newlines are collapsed inside the CSV cell so a spreadsheet row stays
        # one line; articles/doc_<id>.txt keeps the original formatting
        flat_text = re.sub(r"\s+", " ", text).strip()
        truncated = len(flat_text) > max_article_chars
        doc_rows.append(
            [order, doc_id, d.get("_sm_p_bucket_"), d.get("_qa_p_bucket_"),
             round(d.get("_sm_p_", 0.0), 4), round(d.get("_qa_p_", 0.0), 4),
             n_gold, len(text)]
            + [" | ".join(form for form, _ in per_label[l]) for l in LABELS]
            + ["" for _ in LABELS]
            + ["", "", flat_text[:max_article_chars], "yes" if truncated else "no"]
        )

        # entity -> other labels carrying the same surface form in this doc
        label_index: Dict[str, List[str]] = {}
        for l in LABELS:
            for form, _ in per_label[l]:
                label_index.setdefault(form.lower(), []).append(l)

        candidates = [(label, form, count)
                      for label in LABELS for form, count in per_label[label]]

        # Optional within-document cap: a uniform sample of the document's
        # entities. ``weight`` is the inverse inclusion probability, so the
        # scorer can still estimate corpus-level precision and the document's
        # true-positive count without bias. Labels in ``always_labels`` are
        # taken with certainty (weight 1.0) — they are too rare to survive
        # sampling — and the cap applies to the remainder.
        certain = [c for c in candidates if c[0] in (always_labels or [])]
        pool = [c for c in candidates if c[0] not in (always_labels or [])]

        if max_entities_per_doc and len(pool) > max_entities_per_doc:
            picked = sorted(rng.sample(range(len(pool)), max_entities_per_doc))
            drawn = [pool[i] for i in picked]
        else:
            drawn = pool
        pool_weight = len(pool) / len(drawn) if drawn else 1.0

        for label, form, count in certain + drawn:
            row_id += 1
            ev = entity_evidence(text, form)
            others = [l for l in label_index.get(form.lower(), []) if l != label]
            weight = 1.0 if (label in (always_labels or [])) else pool_weight
            ent_rows.append([
                row_id, order, doc_id, label, form,
                count, " | ".join(others),
                ev["in_text"], ev["span_flag"], ev["n_occurrences"],
                round(weight, 4), ev["context"],
                "", "", "", "", "",
            ])

    with open(outdir / "A_docs_recall.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(DOC_HEADER)
        w.writerows(doc_rows)

    with open(outdir / "B_entities_precision.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(ENTITY_HEADER)
        w.writerows(ent_rows)

    return len(doc_rows), len(ent_rows)


README_TEMPLATE = """# NER gold 수기 검증 시트 (R2 #2a)

`original_ner.json`(fine-tuned CyNER, F1 76.23% 주장)의 gold 품질을 √4,940 ≈ {n_docs}건
문서 표본에서 사람이 직접 확인하기 위한 시트입니다. 판정 대상은 **NER 출력**이며,
LLM 요약/QA의 환각 여부는 여기서 판정하지 않습니다(그쪽은 §7 human validation).

## 표본

- 모집단 {n_pop} docs 중 {n_docs} docs (√N 규칙)
- 층화: gpt-4o-mini **summary precision** 사분위 × **QA precision** 사분위 = {n_cells} cell
  (summary precision은 1.0에 질량이 몰려 상위 사분위 경계가 최댓값과 겹치므로 3구간으로 축약)
- seed={seed}, cell round-robin 순서로 기록 → **앞에서부터 일부만 채워도 층화 표본이 유지**됩니다
  (예: 앞 20개 문서만 채워도 {n_cells}개 cell이 모두 포함됨).
- 판정 대상 엔티티 {n_entities}개 (문서 내 중복 표기는 1행으로 합침){cap_note}

## 파일

| 파일 | 용도 |
|---|---|
| **`review.html`** | **판정 GUI — 이것만 브라우저로 열면 됩니다** |
| `B_entities_precision.csv` | 예측 엔티티 1개 = 1행 → **precision** (GUI가 읽고 내보냄) |
| `A_docs_recall.csv` | 문서 1건 = 1행, 누락 엔티티 → **recall** |
| `articles/doc_<id>.txt` | 원문 전문 (GUI 안에도 들어 있음) |

### GUI 사용법

```bash
xdg-open review.html      # 또는 브라우저로 파일 열기
```

- **1** = 정확 → 바로 다음 항목 (오류 유형 입력 없음)
- **2 / 3** = 부분 / 오류 → **q w e r**(스팬 잘림 / 라벨 오류 / 엔티티 아님 /
  원문에 없음) 중 하나를 골라야 다음으로 넘어갑니다
- **← →** 이동, **f** 전문 보기, **c** 올바른 형태, **n** 메모, **u** 판정 취소
- 2번 탭에서 문서별 누락 엔티티를 채우고 **d** 로 확인 완료
- 키 입력마다 브라우저 localStorage에 자동 저장됩니다. 그래도 중간중간
  **JSON 저장**으로 백업해 두세요(브라우저 데이터 삭제 시 날아갑니다)
- 다 하면 **CSV 내보내기** → 받은 `B_entities_precision.csv` / `A_docs_recall.csv`를
  이 폴더에 덮어쓰고 채점 명령을 실행합니다

CSV를 직접 편집해도 결과는 같습니다. 아래 판정 기준은 두 방식 공통입니다.

## B 시트 — precision 판정

미리 계산된 근거 열 (수정하지 마세요):

| 열 | 뜻 |
|---|---|
| `in_text` | 표면형이 원문에 문자열로 존재하는가 |
| `span_flag` | `ok` = 단어 경계로 존재 / `substring_only` = 더 긴 토큰 내부에만 존재(예: `UNC2165` 안의 `165` — 잘린 스팬 신호) / `absent` = 원문에 없음 |
| `n_occurrences` | 원문 내 등장 횟수 |
| `context` | 첫 등장 위치 ±110자 |
| `also_labeled_as` | 같은 문서에서 이 표면형이 함께 달린 다른 라벨 |

`verdict` (필수):

| 값 | 기준 |
|---|---|
| `correct` | 실재하는 엔티티이고, 스팬 경계와 라벨이 모두 맞음 |
| `partial` | 실재하지만 스팬이 잘렸거나(`UNC2`←`UNC2165`) 라벨이 틀림 |
| `incorrect` | 원문에 없거나, 엔티티가 아님(일반 명사 등) |

`error_type` (`partial`/`incorrect`일 때만): `span_boundary` | `wrong_label` |
`not_an_entity` | `not_in_text`
`correct_form`: 올바른 표면형/라벨을 알면 기입 (예: `UNC2165` 또는 `Malware`)
`verdict_b`: 2인 검증 시 두 번째 검수자용 (Cohen's κ 산출). 1인이면 비워둡니다.

## A 시트 — recall 판정

`gold_<Label>` 열에 그 문서에서 NER이 뽑은 엔티티가 이미 들어 있습니다. 원문
(`article_text` 열 또는 `articles/doc_<id>.txt`)을 읽고 **빠진 것만** 대응하는
`missed_<Label>` 열에 ` | ` 로 구분해 적습니다. 없으면 비워 두고, 문서를 다 봤으면
`doc_done` 에 `y` 를 적어 주세요 — 채점 시 "확인했는데 누락 0"과 "아직 안 봄"을
구분하는 데 이 열을 씁니다.

라벨 정의는 CyNER 스키마를 따릅니다: `Malware`, `System`(제품·OS·플랫폼),
`Indicator`(IP·도메인·해시·캠페인 코드), `Vulnerability`(CVE 등), `Organization`.

## 채점

```bash
PYTHONPATH=final_pipeline python -m analyzer.ner_manual_score \\
    --entities {outdir}/B_entities_precision.csv \\
    --docs {outdir}/A_docs_recall.csv \\
    --output notes/ner_manual_validation.md
```

per-label / micro precision(strict·lenient), recall, F1과 Wilson 95% CI를
산출합니다. `verdict_b`가 채워져 있으면 Cohen's κ도 함께 나옵니다.

## 재생성

```bash
PYTHONPATH=final_pipeline python -m analyzer.ner_manual_sheet \\
    --n {n_docs} --seed {seed} --max-entities-per-doc 3 --outdir {outdir}
PYTHONPATH=final_pipeline python -m analyzer.ner_review_app \\
    --entities {outdir}/B_entities_precision.csv \\
    --docs {outdir}/A_docs_recall.csv --output {outdir}/review.html
```

⚠️ 시트를 다시 만들면 `row_id`가 달라져 GUI에 저장된 진행 상황과 어긋납니다.
판정을 시작한 뒤에는 재생성하지 말고, 필요하면 다른 `--outdir` 를 쓰세요.
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Build manual NER validation sheets (R2 #2a)")
    p.add_argument("--original", default="original_ner.json",
                   help="gold NER json ({doc_id: {label: [entities]}})")
    p.add_argument("--gpt-results", default="gpt_results_2503.json",
                   help="gpt_results_*.json supplying original_text (gpt-4o-mini)")
    p.add_argument("--summary-stats", default="notes/doc_stats_summary_gpt4omini.json")
    p.add_argument("--qa-stats", default="notes/doc_stats_qa_gpt4omini.json")
    p.add_argument("--n", type=int, default=70, help="documents to sample (√4940 ≈ 70)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-article-chars", type=int, default=30000,
                   help="article text cap inside the CSV cell (Excel limit is 32,767)")
    p.add_argument("--max-entities-per-doc", type=int, default=0,
                   help="judge at most N entities per document (0 = all); a "
                        "uniform within-doc sample, carried in the weight column")
    p.add_argument("--always-include-labels", default="",
                   help="comma-separated labels exempt from the per-doc cap "
                        "(e.g. Indicator,Vulnerability — too rare to survive it)")
    p.add_argument("--no-articles", action="store_true", help="skip articles/*.txt")
    p.add_argument("--outdir", default="notes/csv/manual_ner_70")
    args = p.parse_args()

    original_ner = json.loads(Path(args.original).read_text(encoding="utf-8"))
    docs = json.loads(Path(args.gpt_results).read_text(encoding="utf-8"))
    summary_stats = json.loads(Path(args.summary_stats).read_text(encoding="utf-8"))
    qa_stats = json.loads(Path(args.qa_stats).read_text(encoding="utf-8"))

    sampled = stratified_sample(
        docs, original_ner, summary_stats, qa_stats, args.n, args.seed
    )

    outdir = Path(args.outdir)
    always = [l.strip() for l in args.always_include_labels.split(",") if l.strip()]
    unknown = [l for l in always if l not in LABELS]
    if unknown:
        p.error(f"--always-include-labels: unknown label(s) {unknown}; pick from {LABELS}")

    n_docs, n_entities = write_sheets(
        sampled, original_ner, outdir, args.max_article_chars, not args.no_articles,
        max_entities_per_doc=args.max_entities_per_doc, seed=args.seed,
        always_labels=always,
    )

    cap_note = ""
    if args.max_entities_per_doc:
        cap_note = (
            f"\n- **문서당 최대 {args.max_entities_per_doc}개**만 판정합니다 "
            f"(문서 내 균등 무작위 추출, seed={args.seed}). `weight` 열이 추출 확률의 "
            f"역수(문서 엔티티 수 ÷ 추출 수)이고, 채점기가 이 가중치로 합산하므로 "
            f"엔티티가 적은 문서에 치우치지 않은 코퍼스 추정치가 나옵니다. "
            f"신뢰구간은 Kish 유효표본크기를 씁니다.\n"
            f"- 대신 **희소 라벨의 표본이 얇아집니다** — Indicator·Vulnerability는 "
            f"라벨별 precision을 보고하기 어려운 수준이 될 수 있습니다. 그 두 라벨의 "
            f"수치가 필요하면 `--always-include-labels Indicator,Vulnerability` "
            f"(희소 라벨만 전수, weight 1.0) 또는 `--max-entities-per-doc 0`(전수)으로 "
            f"다시 생성하세요."
        )
    if args.always_include_labels:
        cap_note += (
            f"\n- `{args.always_include_labels}` 라벨은 cap에서 제외하고 **전수 판정**합니다 "
            f"(weight 1.0). 나머지 라벨에만 문서당 cap이 적용됩니다."
        )

    (outdir / "README.md").write_text(
        README_TEMPLATE.format(
            n_docs=n_docs, n_entities=n_entities, n_pop=len(original_ner),
            n_cells=len({(d["_sm_p_bucket_"], d["_qa_p_bucket_"]) for d in sampled}),
            seed=args.seed, outdir=args.outdir, cap_note=cap_note,
        ),
        encoding="utf-8",
    )

    print(f"[ner_manual_sheet] {n_docs} docs / {n_entities} entity rows → {outdir}")
    cells = sorted({(d["_sm_p_bucket_"], d["_qa_p_bucket_"]) for d in sampled})
    print(f"[ner_manual_sheet] grid cells covered: {len(cells)}")


if __name__ == "__main__":
    main()
