# Moral Meaning Attenuation in Local LLMs

Detecting and quantifying how LLM interpretations can attenuate moral content
in dystopian literature using the Epistemic Fidelity Score (EFS).

## Current Experiment

This repo currently runs a local-model version of the study:

- Models: `gemma3:4b` and `phi4-mini` through Ollama
- Books: 6 dystopian novels
- Passages: 50 per book, 300 total
- Expected interpretation pairs: 300 passages x 2 models = 600 `(S, I)` pairs

The accompanying paper source is aligned with this local-model study. Do not
claim GPT, Claude, Gemini, or other frontier-model behavior unless the code and
experiments are updated to run those models.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Install and start Ollama, then pull the configured models:

```bash
ollama pull gemma3:4b
ollama pull phi4-mini
```

## Run Pipeline

Automatic:

```bash
python run_pipeline.py
```

Manual:

```bash
python src/00b_convert_epubs.py
python src/00_download_books.py
python src/01_extract_passages.py
python src/02_generate_interpretations.py
python src/03_extract_features.py
python src/04_compute_efs.py
python src/05_analyze_results.py
python src/06_validate_metrics.py
python src/07_agreement_analysis.py      # after annotator files exist
```

Resume after a failure:

```bash
python run_pipeline.py --from N
python run_pipeline.py --only N
```

## Books Included

From Project Gutenberg:

- `we`: We, Yevgeny Zamyatin
- `time_machine`: The Time Machine, H. G. Wells
- `iron_heel`: The Iron Heel, Jack London

Manual or EPUB-derived texts:

- `1984`
- `animal_farm`
- `fahrenheit`

Place manual `.txt` files in `data/raw/`, or place EPUBs there and run
`python src/00b_convert_epubs.py`.

## Annotation

After step 4, open `annotations/annotation_template.csv`, fill in
`distortion_score` from 1 to 5, save as `annotations/annotations.csv`, and
rerun step 4 to learn EFS weights through ridge regression.

See `ANNOTATION_GUIDE.md` for scale definitions.

For multi-annotator validation, copy `annotation_template.csv` to files named
`annotations/annotator_NAME.csv`, then run:

```bash
python src/07_agreement_analysis.py
python src/07_agreement_analysis.py --write-consensus
```

## Outputs

- `data/results/interpretations_{model}.csv`: model interpretations
- `data/results/features_all.csv`: source and interpretation feature deltas
- `data/results/efs_scores.csv`: EFS for every pair
- `data/results/summary_statistics.csv`: model summary table
- `data/results/book_summary.csv`: EFS summary by book and model
- `data/results/dimension_summary.csv`: per-dimension summary by model
- `data/results/pairwise_tests.csv`: corrected model comparisons and effect sizes
- `data/results/validation_*.csv`: ablation, baseline, and sensitivity checks
- `data/results/extreme_cases.csv`: highest-distortion examples for qualitative review
- `annotations/agreement_*.csv`: human annotation agreement reports
- `outputs/figures/`: PDF and PNG plots

## Research Status

The implemented code covers the main EFS pipeline, but the research is not
paper-ready until it has generated results, human validation annotations,
baseline comparisons, and ablations. See `RESEARCH_STATUS.md`.

For paper writing, use `PAPER_REVISION_PLAN.md`, `TODO_FLAWLESS_PAPER.md`, and
`QUALITATIVE_ANALYSIS_TEMPLATE.md`.
