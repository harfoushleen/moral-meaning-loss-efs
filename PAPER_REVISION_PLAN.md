# Paper Revision Plan

Use this plan to turn the current project from a framework proposal into a
credible empirical paper.

## Core Decision

Final study identity: local-model study.

- Models: Gemma 3 4B and Phi-4 Mini through Ollama
- Strength: reproducible, cheaper, fully local
- Limitation: cannot claim frontier-model behavior
- Current manuscript source: `paper.tex`

## Recommended Thesis

Use this as the central claim:

> This work proposes and evaluates an Epistemic Fidelity Score for detecting
> directional attenuation of moral meaning in LLM-generated literary
> interpretations.

Avoid this until validated:

> Alignment-trained frontier models systematically erase moral meaning.

## Revised Paper Structure

1. Introduction
2. Related Work
3. Method: Epistemic Fidelity Score
4. Dataset
5. Experimental Setup
6. Results
7. Human Validation
8. Qualitative Analysis
9. Limitations
10. Conclusion

## Section Requirements

### Introduction

Must explain:

- what moral meaning attenuation is
- why it differs from hallucination, refusal, and bias
- why dystopian literature is a good testbed
- what EFS measures
- what evidence the paper contributes 
- Problem not veryyy well motivated= Will not doing it risk something?
- More limitations of previous work

### Method

Must define:

- source passage `S`
- model interpretation `I`
- feature vector `F(S)` and `F(I)`
- deltas `delta_d = f_d(S) - f_d(I)`
- EFS weighted sum
- positive delta as attenuation

### Dataset

Must report:

- six books
- 50 passages per book
- 300 source passages
- selection method: moral keyword ranking
- passage length range
- copyright/public-domain/manual-source note

### Experimental Setup

Must report:

- exact model names
- prompt text
- decoding settings if available
- number of generated interpretations
- local runtime or API environment
- failed calls and retry policy

### Results

Must include:

- model-level EFS mean and standard deviation
- per-dimension delta table
- book-level EFS table
- corrected pairwise tests
- effect sizes
- ablation results
- baseline comparison

### Human Validation

Must include:

- number of annotators
- number of shared rows
- annotation scale
- agreement metrics
- whether consensus or individual scores were used
- ridge-regression weight results

### Qualitative Analysis

Use 3 to 5 examples from:

```text
data/results/extreme_cases.csv
```

For each example, explain:

- what the source says morally
- what the model changed
- which EFS dimension captured it
- whether humans agreed it was distorted

### Limitations

Must say:

- EFS approximates moral meaning through NLP features
- lexicon-based signals can miss irony, context, and literary nuance
- source passage selection is keyword-biased
- local models do not prove frontier-model behavior
- human annotation is subjective
- the metric detects textual attenuation, not model intent

## Tables And Figures Checklist

- Table 1: Dataset summary by book
- Table 2: Model configuration
- Table 3: EFS summary by model
- Table 4: Per-dimension mean deltas
- Table 5: Baseline comparison
- Table 6: Ablation results
- Table 7: Human agreement
- Figure 1: Pipeline diagram
- Figure 2: EFS distribution by model
- Figure 3: Dimension deltas by model
- Figure 4: EFS by book
- Figure 5: Correlation heatmap

## Claims Ladder

Weak but currently defensible:

> The pipeline operationalizes moral meaning attenuation as measurable
> source-interpretation divergence.

Defensible after model outputs:

> The studied models show measurable differences in moral-linguistic
> attenuation across the selected dystopian passages.

Refined after the current local run:

> The studied models show measurable dimensional and book-level variation in
> moral-linguistic attenuation, while aggregate default-weight EFS does not show
> a meaningful overall difference between Gemma 3 4B and Phi-4 Mini.

Defensible after human validation:

> EFS aligns with human judgments of moral meaning attenuation on an annotated
> validation subset.

Only defensible after frontier-model runs:

> Frontier alignment-trained LLMs systematically attenuate moral meaning.

## Final Pre-Submission Checklist

- Paper and code use the same model names.
- Paper and code use the same book list.
- Automated result tables are generated from CSVs in `data/results/`.
- Every current figure has a corresponding script.
- Human annotation agreement is reported.
- Limitations are explicit.
- No unsupported claims about model intent.
- Repository has run instructions.
- Random seeds and sampling choices are documented.
