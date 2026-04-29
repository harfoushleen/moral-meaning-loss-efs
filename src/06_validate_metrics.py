"""
STEP 6: Optional validation analyses for the EFS metric.

Run after 04_compute_efs.py:
    python src/06_validate_metrics.py

Outputs:
  data/results/validation_ablation.csv
  data/results/validation_baselines.csv
  data/results/validation_sensitivity.csv
  data/results/extreme_cases.csv
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from scipy import stats

from config import RESULTS_DIR, DEFAULT_WEIGHTS

FEATURES = ["delta_f1", "delta_f2", "delta_f3", "delta_f4"]
FEATURE_LABELS = {
    "delta_f1": "moral_explicitness",
    "delta_f2": "agency_attribution",
    "delta_f3": "lexical_intensity",
    "delta_f4": "causal_completeness",
}


def load_scores() -> pd.DataFrame:
    path = RESULTS_DIR / "efs_scores.csv"
    if not path.exists():
        raise FileNotFoundError("efs_scores.csv not found. Run 04_compute_efs.py first.")
    return pd.read_csv(path)


def safe_corr(a: pd.Series, b: pd.Series) -> float:
    if a.nunique(dropna=True) < 2 or b.nunique(dropna=True) < 2:
        return np.nan
    return stats.pearsonr(a, b)[0]


def ablation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    base_weights = np.array(DEFAULT_WEIGHTS, dtype=float)

    for omitted in FEATURES:
        keep = [f for f in FEATURES if f != omitted]
        keep_idx = [FEATURES.index(f) for f in keep]
        weights = base_weights[keep_idx]
        weights = weights / weights.sum()

        ablated = df[keep].values @ weights
        rows.append({
            "omitted_dimension": FEATURE_LABELS[omitted],
            "correlation_with_efs": safe_corr(pd.Series(ablated), df["efs"]),
            "mean_absolute_shift": np.mean(np.abs(ablated - df["efs"])),
            "max_absolute_shift": np.max(np.abs(ablated - df["efs"])),
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "validation_ablation.csv", index=False)
    return out


def baseline_analysis(df: pd.DataFrame) -> pd.DataFrame:
    baselines = {
        "moral_explicitness_only": df["delta_f1"],
        "lexical_intensity_only": df["delta_f3"],
        "sentiment_proxy_mean": df[["delta_f1", "delta_f3"]].mean(axis=1),
        "unweighted_all_dimensions": df[FEATURES].mean(axis=1),
    }

    rows = []
    for name, values in baselines.items():
        rows.append({
            "baseline": name,
            "correlation_with_efs": safe_corr(values, df["efs"]),
            "mean": values.mean(),
            "std": values.std(),
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "validation_baselines.csv", index=False)
    return out


def sensitivity_analysis(df: pd.DataFrame, n_samples: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    base_weights = np.array(DEFAULT_WEIGHTS, dtype=float)
    base_weights = base_weights / base_weights.sum()
    models = sorted(df["model_key"].unique())
    original_rank = (
        df.groupby("model_key")["efs"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    same_top = 0
    same_full_rank = 0
    sampled_rows = []

    for _ in range(n_samples):
        noise = rng.normal(loc=1.0, scale=0.15, size=len(base_weights))
        weights = np.clip(base_weights * noise, 0.001, None)
        weights = weights / weights.sum()

        sampled = df[FEATURES].values @ weights
        means = (
            pd.DataFrame({"model_key": df["model_key"], "efs_sample": sampled})
            .groupby("model_key")["efs_sample"]
            .mean()
            .sort_values(ascending=False)
        )
        rank = means.index.tolist()
        same_top += int(rank[0] == original_rank[0])
        same_full_rank += int(rank == original_rank)

    sampled_rows.append({
        "n_models": len(models),
        "n_weight_samples": n_samples,
        "original_rank_highest_distortion_first": " > ".join(original_rank),
        "same_top_model_rate": same_top / n_samples,
        "same_full_rank_rate": same_full_rank / n_samples,
    })

    out = pd.DataFrame(sampled_rows)
    out.to_csv(RESULTS_DIR / "validation_sensitivity.csv", index=False)
    return out


def extreme_cases(df: pd.DataFrame, n_cases: int = 25) -> pd.DataFrame:
    interp_files = sorted(RESULTS_DIR.glob("interpretations_*.csv"))
    if interp_files:
        interps = pd.concat([pd.read_csv(path) for path in interp_files], ignore_index=True)
        cols = ["passage_id", "model_key", "source_passage", "interpretation"]
        merged = df.merge(interps[cols], on=["passage_id", "model_key"], how="left")
    else:
        merged = df.copy()

    out = merged.sort_values("efs", ascending=False).head(n_cases)
    out.to_csv(RESULTS_DIR / "extreme_cases.csv", index=False)
    return out


if __name__ == "__main__":
    print("=== Step 6: Validating EFS metric components ===")
    scores = load_scores()
    print(f"  Loaded {len(scores)} rows.")

    ablation = ablation_analysis(scores)
    print("\n  Ablation analysis")
    print(ablation.round(4))

    baselines = baseline_analysis(scores)
    print("\n  Baseline comparison")
    print(baselines.round(4))

    sensitivity = sensitivity_analysis(scores)
    print("\n  Weight sensitivity")
    print(sensitivity)

    cases = extreme_cases(scores)
    print(f"\n  Saved {len(cases)} high-EFS examples to extreme_cases.csv")
