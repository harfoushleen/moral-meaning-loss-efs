# Research Status

This document tracks where the implementation stands relative to the paper draft
`Meaning_Loss_in_LLMs (3).pdf`.

## Implemented

- Six-book passage dataset with 300 extracted passages.
- Ollama-based interpretation generation for `gemma3:4b` and `phi4-mini`.
- Four EFS dimensions:
  - moral explicitness
  - agency attribution
  - lexical intensity
  - causal completeness
- Signed source-minus-interpretation deltas.
- EFS computation with default weights.
- Optional ridge-regression weight learning from human annotations.
- Model/book summary plots and corrected pairwise tests.
- Optional validation script for ablations, baselines, weight sensitivity, and
  extreme-case extraction.
- Human agreement tooling for multi-annotator validation.
- Paper revision, qualitative analysis, and flawless-paper to-do templates.

## Generated In This Workspace

The local evidence pipeline has been run for the configured two-model study:

- `data/results/interpretations_gemma.csv`
- `data/results/interpretations_phi.csv`
- `data/results/features_all.csv`
- `data/results/efs_scores.csv` with 600 scored `(S, I)` pairs
- `data/results/summary_statistics.csv`
- `data/results/book_summary.csv`
- `data/results/dimension_summary.csv`
- `data/results/pairwise_tests.csv`
- `data/results/validation_ablation.csv`
- `data/results/validation_baselines.csv`
- `data/results/validation_sensitivity.csv`
- `data/results/extreme_cases.csv`
- `outputs/figures/*`

Human validation artifacts have not yet been produced:

- `annotations/annotations.csv`
- `annotations/annotator_*.csv`
- `annotations/agreement_*.csv`

## Paper-Code Alignment

- The current manuscript source (`paper.tex`) is aligned with the local-model
  study: Gemma 3 4B and Phi-4 Mini through Ollama.
- The manuscript and code use the same six-book dataset: `We`,
  `The Time Machine`, `The Iron Heel`, `1984`, `Animal Farm`, and
  `Fahrenheit 451`.
- The manuscript can now report completed automated results from the generated
  CSVs and figures.
- The manuscript treats learned ridge weights as optional and reports default
  weights when annotations are insufficient.

## Strong Next Research Steps

1. Annotate at least 150 to 300 `(source, interpretation)` pairs with 2 or more
   annotators.
2. Report inter-annotator agreement before learning weights.
3. Use `extreme_cases.csv` for qualitative examples in the paper.
4. Add prompt-sensitivity experiments if time permits.
5. Re-run EFS with consensus labels if human agreement is sufficient.

## Current Defensible Claim

The repository contains a reproducible local-model run for measuring directional
attenuation of moral-linguistic signals in model interpretations.

## Claim That Still Needs Evidence

The code does not yet prove that alignment training causes moral meaning
attenuation, nor does it support claims about frontier API models. Those claims
would require additional experiments, generated model outputs, and human
validation.
