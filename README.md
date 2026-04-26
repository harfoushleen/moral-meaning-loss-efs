# Alignment-Induced Meaning Loss in LLMs

Detecting and quantifying how alignment-trained LLMs attenuate moral content
in interpretations of dystopian literature, using the Epistemic Fidelity Score (EFS).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env            # then fill in your API keys
```

## Run Pipeline (in order)

```bash
python src/00_download_books.py       # Download 1984, Animal Farm
python src/01_extract_passages.py     # Extract ~50 passages per book
python src/02_generate_interpretations.py  # Call LLM APIs (~450 calls)
python src/03_extract_features.py     # Extract 4 EFS dimensions
python src/04_compute_efs.py          # Compute EFS scores
python src/05_analyze_results.py      # Generate figures + stats
```

## Adding Books Manually
Place `.txt` files in `data/raw/` and add entries to `src/config.py`.

## Annotation (Optional, improves EFS weights)
After step 4, open `annotations/annotation_template.csv`, fill in
`distortion_score` (1=faithful, 5=heavily distorted) for each row,
save as `annotations/annotations.csv`, then re-run step 4.

## Output
- `data/results/efs_scores.csv` — EFS for every (S, I) pair
- `outputs/figures/` — all plots (PDF + PNG)
- `data/results/summary_statistics.csv` — table for paper