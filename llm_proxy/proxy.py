from typing import Dict
from pathlib import Path
import ujson

# 기존 gpt_response.py를 그대로 활용
from gpt_response import summarize_article, qa_article


class LlmProxy:
    def __init__(self, model_hint: str = 'gpt-3.5-turbo'):
        self.model_hint = model_hint

    def run(self, source_text: str) -> Dict:
        summary = summarize_article(source_text)
        qa = qa_article(source_text)
        return {
            'summary_text': summary,
            'qa_text': qa
        }


def save_results_json(results, output_file: str):
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        ujson.dump(results, f, ensure_ascii=False, indent=2)
