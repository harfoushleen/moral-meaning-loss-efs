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

## Not Yet Run In This Workspace

The repository currently has passage CSVs, but no generated result artifacts:

- `data/results/interpretations_*.csv`
- `data/results/features_all.csv`
- `data/results/efs_scores.csv`
- `outputs/figures/*`
- `annotations/annotations.csv`
- `annotations/annotator_*.csv`
- `annotations/agreement_*.csv`

## Main Mismatches With The Paper Draft

- The paper draft describes GPT-5, Claude 4.6, and Gemini 3. The code currently
  runs Gemma 3 4B and Phi-4 Mini locally through Ollama.
- The paper draft lists `The Hunger Games`, `Divergent`, and `The Maze Runner`.
  The code uses public-domain replacements plus manually supplied Orwell/Bradbury
  texts.
- The paper draft describes clustering and anomaly detection. The code now
  extracts high-EFS extreme cases, but full clustering is still future work.
- The paper draft assumes learned ridge weights. The code falls back to default
  weights unless enough human annotations exist.

## Strong Next Research Steps

1. Decide whether the final paper is a local-model study or a frontier-model
   study, then align both code and manuscript.
2. Generate all interpretations and EFS outputs.
3. Annotate at least 150 to 300 `(source, interpretation)` pairs with 2 or more
   annotators.
4. Report inter-annotator agreement before learning weights.
5. Compare EFS against simple baselines from `06_validate_metrics.py`.
6. Use `extreme_cases.csv` for qualitative examples in the paper.
7. Add prompt-sensitivity experiments if time permits.

## Current Defensible Claim

The code supports a reproducible pipeline for measuring directional attenuation
of moral-linguistic signals in model interpretations.

## Claim That Still Needs Evidence

The code does not yet prove that alignment-trained frontier models systematically
produce moral meaning loss. That claim requires generated model outputs, human
validation, and paper/code alignment.
