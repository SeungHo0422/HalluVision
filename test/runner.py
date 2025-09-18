import argparse
from typing import List, Dict
from tqdm import tqdm

from ..data_scraper.adapter import load_docs_from_text_file
from ..llm_proxy.proxy import LlmProxy, save_results_json


def run_pipeline(input_file: str, output_file: str, splitter: str, start_idx: int, limit: int):
    docs = load_docs_from_text_file(input_file, splitter)
    srcs: List[str] = docs.docs
    if limit is not None and limit > -1:
        srcs = srcs[start_idx:start_idx+limit]
    else:
        srcs = srcs[start_idx:]

    proxy = LlmProxy()
    results: List[Dict] = []

    for i, text in enumerate(tqdm(srcs)):
        try:
            out = proxy.run(text)
            results.append({
                'id': start_idx + i,
                'original_text': text,
                'summary_text': out['summary_text'],
                'qa_text': out['qa_text']
            })
        except Exception as e:
            results.append({
                'id': start_idx + i,
                'original_text': text,
                'error': str(e)
            })

    save_results_json(results, output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', required=True)
    parser.add_argument('--output-file', required=True)
    parser.add_argument('--splitter', default='Share this article:')
    parser.add_argument('--start-idx', type=int, default=0)
    parser.add_argument('--limit', type=int, default=-1)
    args = parser.parse_args()

    run_pipeline(args.input_file, args.output_file, args.splitter, args.start_idx, args.limit)
