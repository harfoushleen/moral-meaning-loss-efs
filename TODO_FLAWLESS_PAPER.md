# To-Do List For A Strong Research Paper

This is the practical checklist. Work top to bottom.

## A. Lock The Study Design

- Decide: local-model study or frontier-model study.
- If local: keep Gemma 3 4B and Phi-4 Mini in the paper.
- If frontier: update code to use GPT/Claude/Gemini APIs before claiming them.
- Align the paper book list with `src/config.py`.
- Align the paper model list with `src/config.py`.

## B. Generate Evidence

- Install Ollama manually.
- Pull `gemma3:4b`.
- Pull `phi4-mini`.
- Run `python src/02_generate_interpretations.py`.
- Run `python src/03_extract_features.py`.
- Run `python src/04_compute_efs.py`.
- Run `python src/05_analyze_results.py`.
- Run `python src/06_validate_metrics.py`.

## C. Validate With Humans

- Copy `annotations/annotation_template.csv` once per annotator.
- Name files `annotations/annotator_NAME.csv`.
- Get at least 2 annotators.
- Best target: 3 annotators rating the same 150 rows.
- Run `python src/07_agreement_analysis.py`.
- If agreement is weak, revise the guide and annotate again.
- Run `python src/07_agreement_analysis.py --write-consensus`.
- Rerun steps 4, 5, and 6 with learned/consensus labels.

## D. Build Results Tables

- Dataset summary by book.
- Model configuration table.
- EFS mean/std/min/max by model.
- EFS mean/std by book.
- Per-dimension deltas by model.
- Pairwise tests with corrected p-values.
- Cohen's d effect sizes.
- Ablation results.
- Baseline comparison.
- Human agreement summary.

## E. Build Qualitative Evidence

- Open `data/results/extreme_cases.csv`.
- Select 3 to 5 strong examples.
- Fill `QUALITATIVE_ANALYSIS_TEMPLATE.md`.
- Include at least one agency-erasure example.
- Include at least one lexical-softening example.
- Include at least one causal-loss example if available.

## F. Rewrite The Paper

- Replace speculative future tense with completed-method language only where
  results exist.
- Remove unsupported claims about GPT/Claude/Gemini unless you run them.
- Add the exact prompt used in `src/config.py`.
- Add the exact model names.
- Add the exact dataset list.
- Add limitations.
- Add human validation method and agreement.

## G. Final Quality Checks

- Every number in the paper points to a generated CSV or figure.
- Every figure is readable in black and white.
- Every claim is either methodological, empirical, or explicitly framed as a
  limitation.
- The conclusion does not claim more than the evidence shows.
- The repo can be cloned and run from README instructions.

## H. Submission Polish

- Use consistent terminology: "moral meaning attenuation" or "moral meaning
  loss"; do not switch randomly.
- Define EFS before using the acronym.
- Make the contribution list concrete.
- Keep examples short and legally safe.
- Include a reproducibility paragraph.
- Include a data/ethics paragraph for copyrighted/manual texts.
