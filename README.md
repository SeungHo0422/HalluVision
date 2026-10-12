# HalluVision

Code for **"Measuring Hallucination in Large Language Models for Cyber Threat Intelligence:
An Exploratory Study"** (Song, Lee, Kim, Choi, Kim; *Computers and Electrical Engineering*, 2026, in press).

HalluVision measures how far LLM-generated summaries and short answers drift from the
cyber-threat-intelligence (CTI) article they were generated from. It has four stages:

1. **data_scraper/** collects articles (Threatpost via Selenium; The Hacker News from a CSV).
2. **llm_proxy/** prompts a model for an abstractive summary and five entity-typed answers
   (OpenAI gpt-3.5-turbo / gpt-4o-mini, or Meta-Llama-3-8B-Instruct).
3. **preprocessor/** extracts `Malware, System, Indicator, Vulnerability, Organization`
   entities from source and output with a fine-tuned CyNER model.
4. **analyzer/** scores each document with three metrics (jina-embeddings-v3 cosine similarity,
   entity answer-matching precision/recall, frequency-based entity similarity), runs the paired
   bootstrap statistics, and holds the revision-stage studies (human validation, NER audit,
   FActScore/QAGS comparison, combined score, ablations).


## Install

```bash
conda env create -f environment.yml        # pinned environment used for the paper (Python 3.11)
conda activate halluvision-revision
# or, minimal:
pip install -r requirements.txt            # torch/transformers/einops only for the cosine metric

export OPENAI_API_KEY=...                  # only needed to regenerate LLM outputs or run the judges
```

For the NER steps install CyNER and download the checkpoint:

```bash
pip install git+https://github.com/aiforsec/CyNER nltk
huggingface-cli download Ricky-dev/halluvision-cyner-xlmr-base --local-dir cyner_ckpt
export CYNER_CKPT=$PWD/cyner_ckpt
```

## Get the data and build a workspace

The analysis scripts read and write a flat *workspace* directory. Download the Mendeley
dataset and lay it out once:

```bash
python scripts/prepare_workspace.py --mendeley ~/Downloads/halluvision_v3 --workspace workspace
export HALLUVISION_WORKSPACE=$PWD/workspace
```

`workspace_map.json` in the download drives the layout; the script also derives the few
files that are not shipped because they duplicate released data.

## Run

End-to-end on new articles (scrape, generate, extract, analyse):

```bash
python main.py --threatpost-urls datasets/threatpost_urls.txt \
               --malware-file datasets/wiki_malware_list.txt \
               --original-ner datasets/ground_truth_ner.json --output-dir pipeline_output
```

Re-evaluate the released outputs (no API calls):

```bash
cd workspace
../scripts/run_analysis.sh                 # summary + Q&A answer matching, Figures 5/6, failed cases
../scripts/run_cosine_stats.sh --skip-embed  # paired bootstrap on the released cosine values
../scripts/run_llama_eval.sh               # Llama-3 vs GPT on the 500-article subset (SKIP_CYNER=1 without the checkpoint)
```

The remaining studies (baselines, human validation, NER audit, ablations) are driven by the
`analyzer` CLIs; each module's docstring documents its inputs, and the dataset's
steps-to-reproduce on Mendeley Data lists which steps need a GPU, the CyNER checkpoint,
or an OpenAI key.

## Layout

```
analyzer/            metrics, statistics, revision studies (see module docstrings)
analyzer/baselines/  FActScore and QAGS runners and comparison
data_scraper/        Threatpost crawler, The Hacker News CSV loader
llm_proxy/           OpenAI and Llama clients, summarizer, Q&A handler
preprocessor/        NER post-processing
scripts/             runners and figure/table scripts that read the workspace
notebooks/           figure notebooks (outputs stripped)
human_eval/          annotation webapp (webapp/) and AWS SAM backend (infra/) for the human validation
datasets/            Threatpost URLs, seed list, and the v2 data files (also on Mendeley)
test/                pytest suite (`python -m pytest test`)
config/, utils/      settings and helpers
```

## Model and file names

File names in the data carry run dates, not model names:
`*_2503*` / `*gpt4omini*` is **gpt-4o-mini**, `*_250427*` / `*gpt35*` is **gpt-3.5-turbo**.
The Mendeley README repeats this table.

## Human-validation webapp

`human_eval/webapp` is a static single-page app; `human_eval/infra` is the serverless backend
(API Gateway, Lambda, S3, DynamoDB) deployed with AWS SAM. `webapp/config.js` ships with
anonymised annotator ids and an empty API URL, which puts the app in offline localStorage
mode for local testing (`cd human_eval/webapp && python -m http.server 8080`). Deployment
steps are in `human_eval/infra/README.md`.

## Citation

```bibtex
@article{song2026halluvision,
  title   = {Measuring Hallucination in Large Language Models for Cyber Threat Intelligence: An Exploratory Study},
  author  = {Song, Seungho and Lee, Donggyu and Kim, Jaehan and Choi, Jiyoung and Kim, Jinwoo},
  journal = {Computers and Electrical Engineering},
  year    = {2026},
  note    = {in press}
}
```

## License

MIT (see `LICENSE`). The fine-tuned checkpoint and the CyNER code it builds on are MIT as well;
the data is CC BY 4.0 with the provenance notes given in the Mendeley README.
