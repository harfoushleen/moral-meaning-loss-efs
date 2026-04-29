# Qualitative Analysis Template

Use this template after running:

```bash
python src/06_validate_metrics.py
```

Open:

```text
data/results/extreme_cases.csv
```

Select 3 to 5 examples with high EFS and clear interpretive distortion.

## Example Template

### Example N: [Book] / [Model]

Passage ID:

```text
[passage_id]
```

Source passage excerpt:

```text
[Keep this short. Use only the minimum excerpt needed for analysis.]
```

Model interpretation excerpt:

```text
[Keep this short. Use only the relevant part of the interpretation.]
```

EFS evidence:

| Dimension | Delta | Interpretation |
|---|---:|---|
| Moral explicitness | [delta_f1] | [What changed?] |
| Agency attribution | [delta_f2] | [What changed?] |
| Lexical intensity | [delta_f3] | [What changed?] |
| Causal completeness | [delta_f4] | [What changed?] |

Human annotation:

```text
Mean distortion score: [score]
Annotator agreement: [agreement detail if available]
```

Analysis paragraph:

```text
The source passage frames [agent] as responsible for [harm/action]. The
interpretation weakens this by [passive voice/euphemism/abstraction/causal
omission]. This is not a factual hallucination; the interpretation remains
topically related, but the moral structure is reduced. EFS captures this through
[dimensions].
```

## Selection Rules

Choose examples that:

- are understandable without too much plot context
- show a specific type of meaning loss
- are not merely low-quality or incoherent generations
- represent more than one model if possible
- represent more than one book if possible

Avoid examples where:

- the source passage is too long to explain
- the model output is truncated
- the issue is factual hallucination rather than moral attenuation
- copyright-sensitive excerpts would need to be too long

## Categories To Look For

Agency erasure:

```text
"The state tortured prisoners" -> "Prisoners experienced hardship"
```

Euphemism:

```text
"execution" -> "consequence"
```

Abstraction:

```text
"guards beat workers" -> "the scene explores power"
```

Causal loss:

```text
"because the regime feared rebellion" -> causal motive omitted
```

False neutrality:

```text
Clear oppression is reframed as competing perspectives or social complexity.
```
