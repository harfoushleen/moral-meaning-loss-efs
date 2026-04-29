# To-Do List For A Strong Research Paper

This is the practical checklist. Work top to bottom.

## A. Lock The Study Design

- Locked: local-model study.
- Keep Gemma 3 4B and Phi-4 Mini in the paper.
- Do not claim GPT/Claude/Gemini/Qwen behavior unless those models are added
  to the code and run.
- Paper book list is aligned with `src/config.py`.
- Paper model list is aligned with `src/config.py`.

## B. Generate Evidence

- Install Ollama manually.
- Pull `gemma3:4b`.
- Pull `phi4-mini`.
- Done in this workspace: `python src/02_generate_interpretations.py`.
- Done in this workspace: `python src/03_extract_features.py`.
- Done in this workspace: `python src/04_compute_efs.py`.
- Done in this workspace: `python src/05_analyze_results.py`.
- Done in this workspace: `python src/06_validate_metrics.py`.

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

- Done: Dataset summary by book.
- Done: Model configuration table.
- Done: EFS mean/std/min/max by model in `data/results/summary_statistics.csv`.
- Done: EFS mean/std by book in `data/results/book_summary.csv`.
- Done: Per-dimension deltas by model in `data/results/dimension_summary.csv`.
- Done: Pairwise tests with corrected p-values in `data/results/pairwise_tests.csv`.
- Done: Cohen's d effect sizes in `data/results/pairwise_tests.csv`.
- Done: Ablation results in `data/results/validation_ablation.csv`.
- Done: Baseline comparison in `data/results/validation_baselines.csv`.
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
- Remove unsupported claims about GPT/Claude/Gemini/Qwen unless you run them.
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

- Use consistent terminology: "moral meaning attenuation"; use "moral meaning
  loss" only when explaining the broader concept.
- Define EFS before using the acronym.
- Make the contribution list concrete.
- Keep examples short and legally safe.
- Include a reproducibility paragraph.
- Include a data/ethics paragraph for copyrighted/manual texts.
