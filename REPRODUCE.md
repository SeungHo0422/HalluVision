# Reproducing the paper

This guide maps each table and figure of *Measuring Hallucination in Large Language Models for
Cyber Threat Intelligence: An Exploratory Study* to the command that regenerates it. Sections are
named as in the paper rather than numbered.

## 0. One-time setup

```bash
conda env create -f environment.yml && conda activate halluvision-revision   # or pip install -r requirements.txt
python scripts/prepare_workspace.py --mendeley <Mendeley v3 download> --workspace workspace
export HALLUVISION_WORKSPACE=$PWD/workspace
cd workspace          # every command below runs here; ../scripts/... and `python -m analyzer...` resolve from the repo
export PYTHONPATH=..  # for `python -m analyzer.<module>` / `python -m data_scraper.<module>`
```

Optional inputs, needed only where marked:

| Mark | Requirement |
|---|---|
| **API** | `OPENAI_API_KEY`; regenerates LLM outputs or runs the FActScore/QAGS judges (cost in the paper's API table) |
| **GPU** | jina-embeddings-v3 (cosine) or Meta-Llama-3-8B-Instruct inference; a 24 GB card suffices |
| **CKPT** | the fine-tuned CyNER checkpoint: `pip install git+https://github.com/aiforsec/CyNER nltk`, `huggingface-cli download <HF_REPO> --local-dir cyner_ckpt`, `export CYNER_CKPT=$PWD/cyner_ckpt` |
| **THN** | The Hacker News source CSV (not redistributed; see `subsets/thehackernews_sample_500.json` in the data for URLs, row indices and text hashes) |

Everything else runs from the released files in a few minutes on a laptop.

## 1. Main corpus: gpt-3.5-turbo vs gpt-4o-mini on 4,940 Threatpost articles

| Paper artefact | Command | Output | Needs |
|---|---|---|---|
| LLM outputs (`gpt_results_*.json`) | `python ../main.py --threatpost-urls ../datasets/threatpost_urls.txt --malware-file ../datasets/wiki_malware_list.txt --original-ner original_ner.json --prompt-version v2` | `pipeline_output/llm_results/` | API |
| Reference entities (`original_ner.json`) | `python ../scripts/run_cyner_ner.py --input gpt_results_2503.json --text-key original_text --output original_ner.json` | entity dict | CKPT, GPU |
| Summary entities | same with `--text-key summary_text` on each `gpt_results_*.json` | `summary_ner.json`, `cyner_sum_results_250427.json` | CKPT, GPU |
| Cosine similarity CDF (Figure "cosine similarity") | `../scripts/run_cosine_stats.sh` (full re-embedding) or `--skip-embed` (reuse released `notes/doc_cosine_*.json`) | `notes/doc_cosine_*.json`, `notes/stats_results_summary_cosine.md` | GPU for re-embedding |
| Figure file | `notebooks/cosine_similarity.ipynb` (reads `notes/doc_cosine_*.json`; style in `scripts/cdf_plot_style.py`) | `figures/figure3_v2.pdf` | |
| Summary answer matching (Figure "precision/recall CDF", Table "answer matching, summarization") | `../scripts/run_analysis.sh --skip-qa` | `figures/figure5.pdf`, `figures/figure6.pdf`, `false_positives_gpt{4o,35}.json` | |
| Publication figures for the summary task | `notebooks/eval_summary_prof.ipynb` | `figures/figure4_{precision,recall}_v2.pdf`, `figures/L1_distance_cdf_v2.pdf` | |
| Frequency-based entity similarity | `python ../scripts/extract_freq_sim.py --original original_ner.json --summary-ner summary_ner.json --output notes/doc_freq_sim_gpt4omini.json` (and `cyner_sum_results_250427.json` for gpt-3.5-turbo) | `notes/doc_freq_sim_*.json` | |
| Q&A answer matching (Table "answer matching, Q&A", failed cases) | `../scripts/run_analysis.sh --skip-summary` | `failed_cases_gpt4omini.csv`, `failed_cases_gpt3.5.csv` | |
| Q&A CDF figures | `notebooks/analysis_qa.ipynb` | `figures/qa_{precision,recall}_cdf_v2.pdf` | |
| Per-document metric tables | `python -m analyzer.extract_doc_stats --task summary --original original_ner.json --predictions summary_ner.json --output notes/doc_stats_summary_gpt4omini.json`; `--task qa --predictions gpt_results_2503.json --normalize-qa --output notes/doc_stats_qa_gpt4omini.json`; same for gpt-3.5-turbo | `notes/doc_stats_*.json` | |
| Paired bootstrap (Table "statistical comparison") | `python -m analyzer.stats --model-a gpt-3.5-turbo --model-b gpt-4o-mini --metrics-a notes/doc_stats_qa_gpt35.json --metrics-b notes/doc_stats_qa_gpt4omini.json --output notes/stats_results_qa.md` (and `doc_stats_summary_*` for the summary task) | `notes/stats_results_{qa,summary}.{md,json}` | |
| Span-matching variants (`span_match_comparison.md`) | `extract_doc_stats` with `--match-mode {exact,token_f1,jaccard,relaxed}` | `notes/doc_stats_qa_*_<mode>.json` | |
| Role-persona ablation | `python -m analyzer.role_ablation --gpt-results gpt_results_2503.json --original-ner original_ner.json --output notes/role_ablation.md --dump-rows notes/role_ablation_rows.json` | `notes/role_ablation*.{md,json}` | API |
| API token usage and cost (Appendix table) | released as `stats/api_token_usage_main_corpus.json`; counts were reconstructed with `tiktoken` from the stored prompts and outputs | | |

## 2. Combined score (Appendix "Combining the Three Metrics")

```bash
python ../scripts/build_entity_freq.py                     # label-count vectors -> notes/combined_scorer_demo/entity_freq_gpt4omini.json
python ../scripts/combined_score_corpus.py                  # z-scored equal-weight score -> notes/combined_score_corpus.{json,md}
python ../scripts/plot_combined_score.py                    # figures/figure_combined_score.pdf
python ../scripts/combined_vs_single.py                     # combined vs single metrics on the human labels -> notes/combined_vs_single.{json,md}
```

## 3. Open-weight model: Llama-3-8B-Instruct on 500 Threatpost articles

```bash
python ../scripts/make_llama_subset.py                      # stratified 500 ids (seed 42) -> pipeline_output/llama/tp_subset_500*.json
python ../run_llama_inference.py --model llama-3-8b-instruct --source pipeline_output/llama/tp_subset_500.json \
       --out-dir pipeline_output/llama --no-system          # GPU; greedy, bf16, 256 new tokens
../scripts/run_llama_eval.sh                                # QA stats, paired bootstrap, cosine (GPU), summary NER (CKPT); SKIP_CYNER=1 to skip NER
python ../scripts/make_llama_csv.py                         # the seven CSV tables -> notes/csv/llama_*.csv
```

Tables of the section "Open-weight Model Evaluation" come from `notes/csv/llama_overview.csv`,
`llama_qa_by_label.csv`, `llama_paired_bootstrap.csv`, `llama_summary_axis.csv`.

## 4. Second source: 500 The Hacker News articles

```bash
python -m data_scraper.thn_csv_loader --csv <THN csv> --n-sample 500 --seed 42 --output pipeline_output/thn_scraped.json   # THN
python ../main.py --pre-scraped pipeline_output/thn_scraped.json --output-dir pipeline_output/thn --prompt-version v2         # API
CUDA_VISIBLE_DEVICES=0 python ../scripts/run_thn_cyner.py                                                                  # CKPT, GPU: gold + summary NER
python ../scripts/thn_qa_precision_recall.py                                        # Table "The Hacker News" (answer matching) -> notes/thn_qa_precision_recall.json
python ../scripts/run_thn_summary_eval.py --pair "gpt-4o-mini (THN)" pipeline_output/thn/thn_original_ner.json pipeline_output/thn/thn_summary_ner_gpt-4o-mini.json \
       --pair "gpt-4o-mini (Threatpost)" original_ner.json summary_ner.json --output notes/thn_summary_eval.md --json-output notes/thn_summary_eval.json
python -m analyzer.cosine_similarity --input pipeline_output/thn/gpt_results_thn_gpt-4o-mini.json --output notes/doc_cosine_thn_gpt-4o-mini.json --pooling cls   # GPU
python -m analyzer.stats --model-a gpt-3.5-turbo --model-b gpt-4o-mini --metrics-a notes/doc_cosine_thn_gpt-3.5-turbo.json \
       --metrics-b notes/doc_cosine_thn_gpt-4o-mini.json --metrics cosine_sim --labels _overall_ --output notes/stats_results_thn_cosine.md
```

## 5. Existing detectors: FActScore and QAGS on 200 documents

```bash
python -m analyzer.baselines.subset --gpt-results gpt_results_2503.json --human-sample notes/human_label_sample.csv --total 200 --seed 42 --output notes/baseline_subset.json
python -m analyzer.baselines.factscore_runner --subset notes/baseline_subset.json --output notes/factscore_scores.json            # API
python -m analyzer.baselines.factscore_runner --subset notes/baseline_subset.json --decomp original --output notes/factscore_scores_origprompt.json
python -m analyzer.baselines.qags_runner --subset notes/baseline_subset.json --output notes/qags_scores.json                      # API
python -m analyzer.baselines.hv_metrics --subset notes/baseline_subset.json --cosine notes/doc_cosine_gpt4omini.json \
       --qa-stats notes/doc_stats_qa_gpt4omini.json --entity-freq notes/combined_scorer_demo/entity_freq_gpt4omini.json --output notes/hv_metrics_baseline_subset.json
python -m analyzer.baselines.baseline_compare --halluvision notes/hv_metrics_baseline_subset.json --factscore notes/factscore_scores.json --qags notes/qags_scores.json --output notes/baseline_comparison.md
python ../scripts/baseline_vs_human.py                                            # disagreement cases (Table "detector vs annotators")
python -m analyzer.baselines.human_vs_baseline --labels notes/human_labels_2026-08-07.csv --factscore notes/factscore_scores.json \
       --qags notes/qags_scores.json --hv notes/hv_metrics_baseline_subset.json --subset notes/baseline_subset.json --output notes/human_vs_baseline.md
python -m analyzer.baselines.dump_logs --factscore notes/factscore_scores.json --qags notes/qags_scores.json --output-dir notes/baseline_logs
```

## 6. Human validation (100 items, 10 annotators)

```bash
python -m analyzer.human_sample --gpt-results gpt_results_2503.json --summary-stats notes/doc_stats_summary_gpt4omini.json \
       --qa-stats notes/doc_stats_qa_gpt4omini.json --n 50 --seed 42 --model-tag gpt-4o-mini --output notes/human_label_sample.csv   # repeat for gpt-3.5-turbo
python -m analyzer.build_items --sample notes/human_label_sample.csv --output ../human_eval/webapp/data/items.json   # blinded payload
python -m analyzer.assign_items --annotators a,b,c,d,e,f,g,h,i,j --redundancy 3 --out ../human_eval/webapp/data/assignments.json
# deploy human_eval/ (see human_eval/infra/README.md), collect, then:
python -m analyzer.merge_human_labels --dump <admin export>.json --annotators a,b,c,d,e,f,g,h,i,j --sample notes/human_label_sample.csv --out notes/human_labels_2026-08-07.csv
python -m analyzer.agreement_multi --labels notes/human_labels_2026-08-07.csv --output notes/human_validation_2026-08-07.md   # Krippendorff's alpha, pairwise kappa
python ../scripts/make_validation_table.py                                        # per-item consensus table
python ../scripts/metric_vs_human.py                                              # metric vs consensus, AUC / Spearman
python ../scripts/human_validation_diagnostics.py                                 # unanimity, adjacent disagreement, collapsed-scale alpha
python ../scripts/plot_human_roc.py                                               # figures/figure_human_roc_{a,b}.pdf
python ../scripts/plot_metric_by_label.py --task summary   (and --task qa)        # figures/figure_metric_by_label_*.pdf
python ../scripts/plot_quartile_rate.py --task summary                            # figures/figure_quartile_rate_*.pdf
```

The released `human_validation/human_labels.csv` is the merged, anonymised sheet, so every step after
`merge_human_labels` runs offline.

## 7. Reliability of the CyNER ground truth (70 articles, two raters)

```bash
python -m analyzer.ner_manual_sheet --original original_ner.json --gpt-results gpt_results_2503.json \
       --summary-stats notes/doc_stats_summary_gpt4omini.json --qa-stats notes/doc_stats_qa_gpt4omini.json --n 70 --seed 42 --outdir notes/csv/manual_ner_70
python -m analyzer.ner_review_app --help                                          # keyboard GUI that produces a progress JSON per rater
python -m analyzer.ner_review_merge --progress notes/csv/manual_ner_70/rater_a_progress.json --outdir notes/csv/manual_ner_70
python -m analyzer.ner_review_merge --progress notes/csv/manual_ner_70/rater_b_adjudicated.json --as-annotator-b --outdir notes/csv/manual_ner_70
python -m analyzer.ner_manual_score --entities notes/csv/manual_ner_70/B_entities_precision.csv --docs notes/csv/manual_ner_70/A_docs_recall.csv \
       --output notes/ner_manual_validation.md --json-output notes/ner_manual_validation.json          # Table "manual NER audit", Cohen's kappa
python -m analyzer.ner_agreement --gold pipeline_output/thn/thn_original_ner.json --pred pipeline_output/thn/thn_gold_baseCyNER.json --output notes/ner_sensitivity_base_vs_finetuned.md
python -m analyzer.error_propagation --help                                       # extractor-error decomposition (Table "extractor decomposition")
```

The released `ner_audit/` directory already contains the filled sheets and both raters' logs.

## 8. Case studies and error taxonomy

```bash
python -m analyzer.qa_error_taxonomy --failed-cases failed_cases_gpt4omini.csv --gpt-results gpt_results_2503.json --output notes/qa_error_taxonomy.md
python -m analyzer.case_study_selector --classified notes/qa_error_taxonomy_rows.json --gpt-results gpt_results_2503.json --output notes/case_studies.md
```

## 9. Verification log

On 2026-10-06 the Mendeley v3 files were laid out with `scripts/prepare_workspace.py` and the
offline steps were re-run in the `environment.yml` environment (Python 3.11, scikit-learn 1.9, NumPy 2.4).

| Command | Result against the paper's stored outputs |
|---|---|
| `run_analysis.sh` | `false_positives_gpt{4o,35}.json` identical up to list order; `failed_cases_*.csv` identical up to the order of entities inside the set-valued columns (6,964 and 8,435 rows); `figures/figure6.pdf` regenerated |
| `run_cosine_stats.sh --skip-embed` | `notes/stats_results_summary_cosine.json` identical |
| `run_llama_eval.sh` (`SKIP_CYNER=1`) | `notes/doc_stats_qa_*_tp500.json`, `notes/stats_llama_vs_*.json`, `notes/llama_summary_eval.json` identical |
| `make_llama_csv.py` | all seven `notes/csv/llama_*.csv` byte-identical |
| `thn_qa_precision_recall.py`, `combined_vs_single.py`, `make_validation_table.py` | identical |
| `build_entity_freq.py`, `combined_score_corpus.py` | identical except floating-point rounding in the last digits |
| `metric_vs_human.py`, `human_validation_diagnostics.py`, `baseline_vs_human.py`, `plot_combined_score.py`, `plot_human_roc.py`, `plot_metric_by_label.py`, `plot_quartile_rate.py` | ran to completion, figures regenerated |
| `python -m pytest test` | 79 passed, 1 warning in 38.65s |

Not re-run (GPU, checkpoint, API key or the THN CSV required): jina-v3 embedding, CyNER extraction,
LLM generation, Llama inference, FActScore/QAGS judge calls. Their outputs are released on Mendeley Data.


## Notes

- Model and file names: `*_2503*` / `*gpt4omini*` is gpt-4o-mini, `*_250427*` / `*gpt35*` is gpt-3.5-turbo.
- `scripts/eval_summary.py` and `scripts/analysis_qa.py` keep their original hard-coded input names;
  `run_analysis.sh` substitutes the paths it is given before executing them. Edit the runner and the
  scripts together.
- Bootstrap outputs are seeded and reproduce exactly; floating-point sums in the per-document files
  can differ in the last one or two digits across NumPy versions.
- The paper's annotator ids `a` to `j` are the only identifiers released.
