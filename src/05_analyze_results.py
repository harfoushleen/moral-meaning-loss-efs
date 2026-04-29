"""
STEP 5: Generate all plots and summary statistics for the paper.
Outputs figures to outputs/figures/

Run: python src/05_analyze_results.py

Fixes applied vs original:
  - Model labels updated to actual API model names (GPT-4o, Gemini 1.5 Pro,
    Claude Sonnet 4.6) instead of fabricated names from the paper.
  - Pairwise t-tests now apply Bonferroni correction for multiple comparisons
    (3 comparisons). Raw p-values alone would inflate false positive rate.
  - Cohen's d effect size is reported alongside every t-test so statistical
    significance is not confused with practical significance.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from config import RESULTS_DIR, FIG_DIR, MODEL_DISPLAY_NAMES

plt.rcParams.update({
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "figure.dpi":      150,
    "savefig.bbox":    "tight",
    "savefig.dpi":     300,
})

MODEL_COLORS = {
    "claude": "#e07b39",
    "gpt":    "#4a90d9",
    "gemini": "#5cb85c",
}


def label(model_key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def load_data() -> pd.DataFrame:
    efs_file = RESULTS_DIR / "efs_scores.csv"
    if not efs_file.exists():
        raise FileNotFoundError("efs_scores.csv not found. Run 04_compute_efs.py first.")
    return pd.read_csv(efs_file)


# ── Plot 1: EFS Distribution per Model ─────────────────────────────────────

def plot_efs_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))

    for model_key, color in MODEL_COLORS.items():
        subset = df[df["model_key"] == model_key]["efs"]
        if subset.empty:
            continue
        subset.plot.kde(ax=ax, color=color, label=label(model_key), linewidth=2)
        ax.axvline(subset.mean(), color=color, linestyle="--", alpha=0.7, linewidth=1)

    ax.set_xlabel("Epistemic Fidelity Score (EFS)\n(higher = more distortion)")
    ax.set_ylabel("Density")
    ax.set_title("EFS Distribution by Model")
    ax.legend()
    ax.grid(alpha=0.3)

    _save(fig, "fig1_efs_distribution")


# ── Plot 2: Per-Dimension Delta Comparison ──────────────────────────────────

def plot_dimension_comparison(df: pd.DataFrame):
    dims       = ["delta_f1", "delta_f2", "delta_f3", "delta_f4"]
    dim_labels = ["Moral\nExplicitness", "Agency\nAttribution",
                  "Lexical\nIntensity", "Causal\nCompleteness"]

    models = [m for m in MODEL_COLORS if m in df["model_key"].unique()]
    x      = np.arange(len(dims))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model_key in enumerate(models):
        subset = df[df["model_key"] == model_key]
        means  = [subset[d].mean() for d in dims]
        stds   = [subset[d].std()  for d in dims]
        ax.bar(x + i * width, means, width,
               yerr=stds, capsize=3,
               label=label(model_key),
               color=MODEL_COLORS[model_key], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels(dim_labels)
    ax.set_ylabel("Mean Delta (Source − Interpretation)")
    ax.set_title("Mean Distortion per Dimension by Model")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    _save(fig, "fig2_dimension_comparison")


# ── Plot 3: EFS by Book ─────────────────────────────────────────────────────

def plot_efs_by_book(df: pd.DataFrame):
    books  = sorted(df["book"].unique())
    models = [m for m in MODEL_COLORS if m in df["model_key"].unique()]
    x      = np.arange(len(books))
    width  = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model_key in enumerate(models):
        subset = df[df["model_key"] == model_key]
        means  = [subset[subset["book"] == b]["efs"].mean() for b in books]
        ax.bar(x + i * width, means, width,
               label=label(model_key),
               color=MODEL_COLORS[model_key], alpha=0.85)

    ax.set_xticks(x + width)
    ax.set_xticklabels([b.replace("_", " ").title() for b in books])
    ax.set_ylabel("Mean EFS")
    ax.set_title("Mean EFS by Source Book and Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    _save(fig, "fig3_efs_by_book")


# ── Plot 4: Correlation Heatmap ─────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame):
    cols   = ["delta_f1", "delta_f2", "delta_f3", "delta_f4", "efs"]
    labels = ["Δ Moral", "Δ Agency", "Δ Lexical", "Δ Causal", "EFS"]

    corr         = df[cols].corr()
    corr.index   = labels
    corr.columns = labels

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Correlation Between EFS Dimensions")

    _save(fig, "fig4_correlation_heatmap")


# ── Statistics Table ────────────────────────────────────────────────────────

def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Compute Cohen's d effect size between two independent samples."""
    pooled_std = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return (a.mean() - b.mean()) / (pooled_std + 1e-9)


def print_stats_table(df: pd.DataFrame):
    print("\n  === Full Statistics Table ===")

    dims       = ["delta_f1", "delta_f2", "delta_f3", "delta_f4", "efs"]
    dim_labels = ["Δ Moral Expl.", "Δ Agency Attr.",
                  "Δ Lexical Int.", "Δ Causal Compl.", "EFS"]

    for model_key in sorted(df["model_key"].unique()):
        subset = df[df["model_key"] == model_key]
        print(f"\n  {label(model_key)} (n={len(subset)})")
        print(f"  {'Dimension':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*56}")
        for dim, lbl in zip(dims, dim_labels):
            col = subset[dim]
            print(f"  {lbl:<22} {col.mean():>8.4f} {col.std():>8.4f} "
                  f"{col.min():>8.4f} {col.max():>8.4f}")

    # Pairwise t-tests with Bonferroni correction and Cohen's d
    models = sorted(df["model_key"].unique())
    pairs  = [(models[i], models[j])
              for i in range(len(models)) for j in range(i + 1, len(models))]
    n_comparisons = len(pairs)

    print(f"\n  === Pairwise t-tests on EFS "
          f"(Bonferroni-corrected, {n_comparisons} comparisons) ===")

    for m1, m2 in pairs:
        a = df[df["model_key"] == m1]["efs"]
        b = df[df["model_key"] == m2]["efs"]

        t_stat, p_raw       = stats.ttest_ind(a, b)
        p_corrected         = min(p_raw * n_comparisons, 1.0)   # Bonferroni
        d                   = cohens_d(a, b)

        sig = ("***" if p_corrected < 0.001
               else "**"  if p_corrected < 0.01
               else "*"   if p_corrected < 0.05
               else "ns")

        print(f"  {label(m1)} vs {label(m2)}: "
              f"t={t_stat:.3f}, p_raw={p_raw:.4f}, "
              f"p_bonf={p_corrected:.4f} {sig}, d={d:.3f}")

    # Save summary CSV
    out_file = RESULTS_DIR / "summary_statistics.csv"
    summary  = df.groupby("model_key")[dims].agg(["mean", "std", "min", "max"])
    summary.to_csv(out_file)
    print(f"\n  Saved summary statistics to {out_file}")


# ── Helper ──────────────────────────────────────────────────────────────────

def _save(fig, name: str):
    pdf = FIG_DIR / f"{name}.pdf"
    png = FIG_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    print(f"  Saved {pdf.name}")


if __name__ == "__main__":
    print("=== Step 5: Generating analysis and figures ===\n")
    df = load_data()
    print(f"  Loaded {len(df)} rows across {df['model_key'].nunique()} models.\n")

    plot_efs_distribution(df)
    plot_dimension_comparison(df)
    plot_efs_by_book(df)
    plot_correlation_heatmap(df)
    print_stats_table(df)

    print(f"\nAll figures saved to {FIG_DIR}")

