# HalluVision Pipeline

This is an integrated CTI (Cyber Threat Intelligence) domain data analysis pipeline for academic research.

## Overview

This pipeline consists of the following 4 stages:

1. **Data Scraper**: Collects malware information from various sources
2. **LLM Proxy**: Handles summarization and question answering (QA) using GPT
3. **Preprocessor**: Preprocesses data for analysis
4. **Analyzer**: Analyzes results and calculates evaluation metrics

## Quick Start

### Installation

```bash
cd final_pipeline
pip install -r requirements.txt
export OPENAI_API_KEY="your-openai-api-key"
```

### Quickstart

```bash
# Overall End-to-End pipeline
python3 main.py \
--original-ner datasets/ground_truth_ner.json \
--threatpost-urls datasets/threatpost_urls.txt \
--malware-file datasets/wiki_malware_list.txt \
```


## Project Structure

```
final_pipeline/
├── data_scraper/           # 데이터 수집 모듈
│   ├── crawler/           # 다양한 소스 크롤러
│   ├── main.py           # 데이터 스크래퍼 메인
│   └── adapter.py        # 데이터 형식 어댑터
├── llm_proxy/            # LLM 처리 모듈
│   ├── gpt_client.py     # OpenAI API 클라이언트
│   ├── summarizer.py     # 요약 처리
│   └── qa_handler.py     # QA 처리
├── preprocessor/         # 전처리 모듈
│   ├── summary_preprocessor.py
│   └── qa_preprocessor.py
├── analyzer/            # 분석 모듈
│   ├── summary_analyzer.py
│   ├── qa_analyzer.py
│   └── visualization.py
├── config/              # 설정
│   └── settings.py
├── utils/               # 유틸리티
│   └── helpers.py
├── main.py             # 메인 파이프라인
└── requirements.txt    # 의존성
```

## 📈 Output Structure
```
output_directory/
├── scraped_data/
│   ├── scraped_data.json         # 수집된 원시 데이터
│   └── intermediate/             # 중간 결과
├── llm_results/
│   ├── summary_results.json      # 요약 결과
│   ├── qa_results.json          # QA 결과
│   └── gpt_results_*.json       # 호환성을 위한 레거시 형식
├── analysis/
│   ├── analysis_results.json    # 분석 결과
│   └── failed_cases_*.csv       # 실패 사례 분석
├── figures/
│   ├── qa_performance_heatmap.pdf
│   ├── failure_analysis.pdf
│   └── model_comparison.pdf
└── PIPELINE_REPORT.md           # 최종 리포트
```

## Configuration

### 환경 변수

```bash
# 필수
export OPENAI_API_KEY="your-api-key"

# 선택사항
export DEFAULT_MODEL="gpt-3.5-turbo"
export ALTERNATIVE_MODEL="gpt-4o-mini"
```

### 설정 파일

`config/settings.py`에서 다양한 설정을 조정할 수 있습니다:

- 모델 설정
- 요약 파라미터 (단어 수, 온도 등)
- QA 라벨 정의
- 기본 경로 설정