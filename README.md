# Alignment-Induced Meaning Loss in LLMs

Detecting and quantifying how alignment-trained LLMs attenuate moral content
in interpretations of dystopian literature, using the Epistemic Fidelity Score (EFS).

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # **REQUIRED** — downloads the spacy model
cp .env.example .env            # then fill in your API keys
```

## Run Pipeline (in order)

**Option A: Automatic (recommended)**
```bash
python run_pipeline.py        # Runs all steps with error-checking
```

**Option B: Manual** (if you have EPUBs, do this first)
```bash
# If using EPUB books, convert them first:
python src/00b_convert_epubs.py    # Convert .epub → .txt (one-time)

# Then run the pipeline:
python src/00_download_books.py       # Download books from Gutenberg
python src/01_extract_passages.py     # Extract ~50 passages per book
python src/02_generate_interpretations.py  # Call LLM APIs (~900 calls)
python src/03_extract_features.py     # Extract 4 EFS dimensions
python src/04_compute_efs.py          # Compute EFS scores
python src/05_analyze_results.py      # Generate figures + stats
```

If a step fails, fix the error and resume with:
```bash
python run_pipeline.py --from N       # Resume from step N (0–5)
python run_pipeline.py --only N       # Run only step N
```

## Adding Books Manually
Place `.txt` files in `data/raw/` and add entries to `src/config.py`.

## Books Included

The pipeline uses **6 dystopian novels**:

**From Project Gutenberg (3 books):**
- We (Zamyatin, ID: 61963)
- The Time Machine (H.G. Wells, ID: 12163)
- The Iron Heel (Jack London, ID: 1164)

**From personal EPUBs (3 books):**
- 1984 (Orwell)
- Animal Farm (Orwell)
- Fahrenheit 451 (Bradbury)

Total: **50 passages × 6 books × 3 models = 900 (S, I) pairs**

### Adding Your EPUB Files

1. **Place your EPUB files** in `data/raw/` with these names:
   ```
   1984.epub
   animal_farm.epub
   fahrenheit.epub
   ```

2. **Convert EPUBs to plain text:**
   ```bash
   python src/00b_convert_epubs.py
   ```
   This creates `.txt` files from your EPUBs (one-time).

3. **Continue with the pipeline:**
   ```bash
   python run_pipeline.py
   ```
   Step 0 will skip Gutenberg downloads for EPUB books (since `.txt` files already exist).

## Annotation (Optional, improves EFS weights)
After step 4, open `annotations/annotation_template.csv`, fill in
`distortion_score` (1=faithful, 5=heavily distorted) for each row,
save as `annotations/annotations.csv`, then re-run step 4.

See **ANNOTATION_GUIDE.md** for detailed scale definitions and examples.

## Model Names in Paper vs. Code

The analysis uses real model names for reproducibility:
- **Code**: `gpt-4o`, `claude-sonnet-4-6`, `gemini-1.5-pro`
- **Paper**: Referred to as GPT-5, Claude 4.6, Gemini 3 (for consistency with naming conventions in the lit.)

The `MODEL_LABELS` dict in `src/05_analyze_results.py` controls plot/table labels.

## Output
- `data/results/efs_scores.csv` — EFS for every (S, I) pair
- `outputs/figures/` — all plots (PDF + PNG)
- `data/results/summary_statistics.csv` — table for paper