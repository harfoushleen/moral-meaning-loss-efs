# Team To-Do List

Use this checklist to finish the paper and validation work after the completed
local-model pipeline run.

## Current State

- The local experiment has been run for Gemma 3 4B and Phi-4 Mini.
- The dataset contains 300 passages across six dystopian novels.
- The run produced 600 source-interpretation pairs.
- EFS scores, summary statistics, book summaries, dimension summaries,
  pairwise tests, ablations, baselines, sensitivity analysis, extreme cases,
  and figures have been generated.
- The automated aggregate EFS comparison does not show a meaningful overall
  difference between Gemma 3 4B and Phi-4 Mini.
- The strongest remaining evidence gap is human validation.

## 1. Human Annotation

Owner suggestion: all team members.

1. Open `annotations/annotation_template.csv`.
2. Create one copy per annotator using the exact naming format:
   `annotations/annotator_NAME.csv`.
3. Each annotator should rate the same shared rows.
4. Minimum defensible target: 30 shared rows across at least 2 annotators.
5. Strong target: 150 shared rows across 3 annotators.
6. Best target if time permits: 300 shared rows across 3 annotators.
7. Fill only the human-rating columns described in `ANNOTATION_GUIDE.md`.
8. Do not edit passage IDs, model keys, source passages, or interpretations.
9. After annotation, run:

```bash
python src/07_agreement_analysis.py
```

10. If agreement is acceptable, create consensus labels:

```bash
python src/07_agreement_analysis.py --write-consensus
```

11. Rerun EFS scoring and validation if consensus labels are produced:

```bash
python src/04_compute_efs.py
python src/05_analyze_results.py
python src/06_validate_metrics.py
```

## 2. Qualitative Analysis

Owner suggestion: writing lead plus one reviewer.

1. Open `data/results/extreme_cases.csv`.
2. Select 3 to 5 examples with high EFS scores.
3. Include at least:
   - one agency-erasure or agency-blurring example
   - one lexical-softening example
   - one causal-loss example, if available
4. For each selected example, fill `QUALITATIVE_ANALYSIS_TEMPLATE.md`.
5. Keep source quotations short and legally safe.
6. For each case, explain:
   - what the source passage says morally
   - what the model interpretation changes
   - which EFS dimension captures the change
   - whether human annotators agreed after validation
7. Add the finalized examples to the Qualitative Analysis section of
   `paper.tex`.

## 3. Paper Results Integration

Owner suggestion: paper lead.

1. Verify every number in `paper.tex` against these files:
   - `data/results/summary_statistics.csv`
   - `data/results/book_summary.csv`
   - `data/results/dimension_summary.csv`
   - `data/results/pairwise_tests.csv`
   - `data/results/validation_ablation.csv`
   - `data/results/validation_baselines.csv`
   - `data/results/validation_sensitivity.csv`
2. Add or refine tables for:
   - model-level EFS summary
   - per-dimension deltas
   - book-level EFS summary
   - ablation results
   - baseline comparison
   - human agreement, after annotation
3. Make sure the paper does not claim that one model is globally worse based
   on aggregate EFS, because the current pairwise test is not significant.
4. Emphasize dimensional and book-level variation instead.
5. Keep the frontier-model limitation explicit: these results apply only to
   local Gemma 3 4B and Phi-4 Mini runs.

## 4. Figures

Owner suggestion: methods/results lead.

1. Review all figures in `outputs/figures/`.
2. Confirm that each figure is readable in black and white.
3. Confirm that axis labels are clear and match the terminology in the paper.
4. Decide which figures belong in the main paper and which can be appendix
   material.
5. Reference the chosen figures in `paper.tex`.

## 5. Reproducibility Checks

Owner suggestion: technical lead.

1. Clone the repository into a clean folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

3. Confirm Ollama is installed and the models are available:

```bash
ollama pull gemma3:4b
ollama pull phi4-mini
```

4. Run the pipeline:

```bash
python run_pipeline.py
```

5. Confirm the regenerated files match the expected row counts:
   - `data/results/efs_scores.csv`: 600 rows
   - `annotations/annotation_template.csv`: one row per scored pair
   - `outputs/figures/`: PNG and PDF outputs
6. Note any environment-specific issues in `README.md`.

## 6. Final Claim Check

Owner suggestion: whole team.

Before submission, confirm:

1. The paper says "moral meaning attenuation" consistently.
2. "Moral meaning loss" is used only when explaining the broader concept.
3. EFS is defined before the acronym is used.
4. Every empirical claim points to a generated artifact.
5. Every limitation is explicit.
6. The paper does not claim model intent.
7. The paper does not claim alignment training caused the observed patterns.
8. The paper does not generalize to GPT, Claude, Gemini, or other frontier
   models.
9. The conclusion matches the actual evidence.

## 7. Submission Polish

Owner suggestion: final editor.

1. Compile `paper.tex` with LaTeX.
2. Fix citation warnings and missing references.
3. Check table widths in the conference format.
4. Check figure placement and captions.
5. Proofread for repeated wording and tense consistency.
6. Add human-validation results once available.
7. Run one final clone-and-run reproducibility check.
