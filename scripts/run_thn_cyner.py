"""THN gold + summary NER with the paper's fine-tuned CyNER checkpoint (R2 #5).

Replicates the recipe of the original gold run (`Inference_Based_All_Data.py` on
the GPU box): nltk sentence split -> cyner.CyNER(.ckpt, use_heuristic=True,
flair_model='ner') -> keep the 5 target types.

Two post-steps were reverse-engineered from `original_ner.json` by re-running the
checkpoint on Threatpost articles 31/33/34 and diffing: the published gold is the
raw prediction list with (a) order-preserving dedup and (b) entities of length <= 2
dropped. Both are applied here so THN gold is produced by the same rule.

Run (from the workspace):
    CYNER_CKPT=<path to the released checkpoint> CUDA_VISIBLE_DEVICES=0 python scripts/run_thn_cyner.py
"""
import os, json, argparse, time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cyner
import nltk

CKPT = os.environ.get('CYNER_CKPT', 'cyner_ckpt')  # fine-tuned CyNER checkpoint (Hugging Face release)
ROOT = os.environ.get('HALLUVISION_WORKSPACE', os.getcwd())  # data workspace
OUT_DIR = os.path.join(ROOT, 'pipeline_output', 'thn')
TARGETS = ['System', 'Malware', 'Organization', 'Indicator', 'Vulnerability']


def clean(ents):
    """Order-preserving dedup + drop <=2-char fragments (matches original_ner.json)."""
    seen, out = set(), []
    for e in ents:
        if len(e) <= 2 or e in seen:
            continue
        seen.add(e)
        out.append(e)
    return out


def extract(model, text):
    raw = {t: [] for t in TARGETS}
    for sent in nltk.sent_tokenize(text):
        if not sent.strip():
            continue
        for ent in model.get_entities(sent):
            if ent.entity_type in TARGETS:
                raw[ent.entity_type].append(ent.text)
    return {t: clean(raw[t]) for t in TARGETS}


def run_set(model, docs, text_key, out_path, label):
    """docs: list of dicts carrying an integer `id` and `text_key`."""
    results = {}
    t0 = time.time()
    for n, d in enumerate(docs, 1):
        doc_id = d['id']
        results[str(doc_id)] = {'id': doc_id, **extract(model, d[text_key] or '')}
        if n % 25 == 0 or n == len(docs):
            el = time.time() - t0
            print(f'[{label}] {n}/{len(docs)}  {el:.0f}s  eta {el/n*(len(docs)-n):.0f}s',
                  flush=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f'[{label}] -> {out_path}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=CKPT, help='fine-tuned CyNER checkpoint dir (env CYNER_CKPT)')
    ap.add_argument('--limit', type=int, default=0, help='debug: first N docs only')
    ap.add_argument('--skip-gold', action='store_true')
    ap.add_argument('--skip-summary', action='store_true')
    args = ap.parse_args()

    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    model = cyner.CyNER(transformer_model=args.ckpt, use_heuristic=True, flair_model='ner')

    if not args.skip_gold:
        # source docs — order in thn_scraped.json defines id 0..499 (same as the LLM run)
        scraped = json.load(open(os.path.join(ROOT, 'pipeline_output', 'thn_scraped.json')))
        docs = [{'id': i, 'combined_content': a['combined_content']}
                for i, a in enumerate(scraped)]
        if args.limit:
            docs = docs[:args.limit]
        run_set(model, docs, 'combined_content',
                os.path.join(OUT_DIR, 'thn_original_ner.json'), 'gold')

    if not args.skip_summary:
        for tag in ('gpt-4o-mini', 'gpt-3.5-turbo'):
            src = os.path.join(OUT_DIR, f'gpt_results_thn_{tag}.json')
            docs = json.load(open(src))
            if args.limit:
                docs = docs[:args.limit]
            run_set(model, docs, 'summary_text',
                    os.path.join(OUT_DIR, f'thn_summary_ner_{tag}.json'), f'summary:{tag}')


if __name__ == '__main__':
    main()
