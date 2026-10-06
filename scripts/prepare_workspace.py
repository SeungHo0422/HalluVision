#!/usr/bin/env python3
"""Lay out the Mendeley Data download as the workspace the analysis scripts expect.

    python scripts/prepare_workspace.py --mendeley ~/Downloads/halluvision_v3 --workspace workspace
    cd workspace && HALLUVISION_WORKSPACE=$PWD ../scripts/run_analysis.sh

The Mendeley package carries `workspace_map.json`, mapping each released file to
the legacy relative path(s) used by the scripts (for example
`llm_outputs/llm_results_gpt4o_mini_2503.json` -> `gpt_results_2503.json`).
Files are symlinked by default; pass --copy to copy instead.

Also derives files that are not shipped because they duplicate released data:
  pipeline_output/llama/tp_subset_500.json            (the 500 Threatpost docs of the Llama run)
  pipeline_output/llama/gpt_subset_{gpt4omini,gpt35}.json, summary_ner_subset_*.json
  figures/, notes/csv/                                 (output directories)
`pipeline_output/thn_scraped.json` (The Hacker News article text) is NOT
redistributed; rebuild it with data_scraper/thn_csv_loader.py --n-sample 500 --seed 42
from the source CSV and check the hashes in subsets/thehackernews_sample_500.json.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def place(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mendeley", required=True, help="unpacked Mendeley Data v3 directory")
    ap.add_argument("--workspace", default="workspace")
    ap.add_argument("--copy", action="store_true", help="copy instead of symlink")
    args = ap.parse_args()

    md = Path(args.mendeley).resolve()
    ws = Path(args.workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    wsmap = json.loads((md / "workspace_map.json").read_text(encoding="utf-8"))

    n = 0
    for rel, targets in wsmap.items():
        src = md / rel
        if not src.exists():
            print(f"  missing in download: {rel}")
            continue
        for t in targets:
            place(src, ws / t, args.copy)
            n += 1
    print(f"placed {n} files under {ws}")

    for d in ("figures", "notes/csv", "pipeline_output/llama", "pipeline_output/thn", ".cosine_ckpt"):
        (ws / d).mkdir(parents=True, exist_ok=True)

    # Llama subset documents = the gpt-4o-mini records restricted to the released ids
    ids_p = ws / "pipeline_output/llama/tp_subset_500_ids.json"
    src_p = ws / "gpt_results_2503.json"
    out_p = ws / "pipeline_output/llama/tp_subset_500.json"
    if ids_p.exists() and src_p.exists() and not out_p.exists():
        ids = {str(i) for i in json.loads(ids_p.read_text(encoding="utf-8"))}
        rows = [r for r in json.loads(src_p.read_text(encoding="utf-8")) if str(r["id"]) in ids]
        out_p.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"derived {out_p.relative_to(ws)} ({len(rows)} docs)")

    # GPT runs and summary NER restricted to the same 500 ids (step 1 of run_llama_eval.sh)
    if ids_p.exists():
        ids = {str(i) for i in json.loads(ids_p.read_text(encoding="utf-8"))}
        for tag, src_rel in (("gpt4omini", "gpt_results_2503.json"), ("gpt35", "gpt_results_250427.json")):
            src, dst = ws / src_rel, ws / f"pipeline_output/llama/gpt_subset_{tag}.json"
            if src.exists() and not dst.exists():
                rows = [r for r in json.loads(src.read_text(encoding="utf-8")) if str(r["id"]) in ids]
                dst.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
                print(f"derived {dst.relative_to(ws)} ({len(rows)} docs)")
        for tag, src_rel in (("gpt4omini", "summary_ner.json"), ("gpt35", "cyner_sum_results_250427.json")):
            src, dst = ws / src_rel, ws / f"pipeline_output/llama/summary_ner_subset_{tag}.json"
            if src.exists() and not dst.exists():
                d = json.loads(src.read_text(encoding="utf-8"))
                sub = {k: v for k, v in d.items() if k in ids}
                dst.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
                print(f"derived {dst.relative_to(ws)} ({len(sub)} docs)")

    print("note: pipeline_output/thn_scraped.json is not redistributed (see --help).")


if __name__ == "__main__":
    main()
