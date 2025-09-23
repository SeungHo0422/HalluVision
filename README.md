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

## Output Structure
```
output_directory/
├── scraped_data/
│   ├── scraped_data.json
│   └── intermediate/
├── llm_results/
│   ├── summary_results.json 
│   ├── qa_results.json
│   ├── gpt_results_3.5.json
│   └── gpt_results_4o_mini.json
├── analysis/
│   ├── analysis_results.json
│   └── failed_cases_*.csv
├── figures/
│   ├── qa_performance_heatmap.pdf
│   ├── failure_analysis.pdf
│   └── model_comparison.pdf
└── PIPELINE_REPORT.md
```

## Configuration

### Env file example

```bash
export OPENAI_API_KEY="your-api-key"
export DEFAULT_MODEL="gpt-3.5-turbo"
export ALTERNATIVE_MODEL="gpt-4o-mini"
```
