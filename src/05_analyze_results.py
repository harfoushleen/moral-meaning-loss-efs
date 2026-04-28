"""
STEP 5: Generate all plots and summary statistics for the paper.
Outputs figures to outputs/figures/

Run: python src/05_analyze_results.py
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from config import RESULTS_DIR, FIG_DIR

# Paper-quality plot settings
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

MODEL_COLORS = {
    "claude": "#e07b39",
    "gpt":    "#4a90d9",
    "gemini": "#5cb85c",
}
MODEL_LABELS = {
    "claude": "Claude 4.6",
    "gpt":    "GPT-5",
    "gemini": "Gemini 3",
}


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
        subset.plot.kde(ax=ax, color=color,
                        label=MODEL_LABELS.get(model_key, model_key),
                        linewidth=2)
        ax.axvline(subset.mean(), color=color, linestyle="--", alpha=0.7, linewidth=1)
    
    ax.set_xlabel("Epistemic Fidelity Score (EFS)\n(higher = more distortion)")
    ax.set_ylabel("Density")
    ax.set_title("EFS Distribution by Model")
    ax.legend()
    ax.grid(alpha=0.3)
    
    out = FIG_DIR / "fig1_efs_distribution.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {out}")


# ── Plot 2: Per-Dimension Delta Comparison ──────────────────────────────────

def plot_dimension_comparison(df: pd.DataFrame):
    dims = ["delta_f1", "delta_f2", "delta_f3", "delta_f4"]
    dim_labels = ["Moral\nExplicitness", "Agency\nAttribution",
                  "Lexical\nIntensity", "Causal\nCompleteness"]
    
    models = [m for m in MODEL_COLORS if m in df["model_key"].unique()]
    x = np.arange(len(dims))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(9, 5))
    
    for i, model_key in enumerate(models):
        subset = df[df["model_key"] == model_key]
        means  = [subset[d].mean() for d in dims]
        stds   = [subset[d].std()  for d in dims]
        
        ax.bar(x + i * width, means, width,
               yerr=stds, capsize=3,
               label=MODEL_LABELS.get(model_key, model_key),
               color=MODEL_COLORS[model_key], alpha=0.85)
    
    ax.set_xticks(x + width)
    ax.set_xticklabels(dim_labels)
    ax.set_ylabel("Mean Delta (Source − Interpretation)")
    ax.set_title("Mean Distortion per Dimension by Model")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    out = FIG_DIR / "fig2_dimension_comparison.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {out}")


# ── Plot 3: EFS by Book ─────────────────────────────────────────────────────

def plot_efs_by_book(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(9, 5))
    
    books = sorted(df["book"].unique())
    models = [m for m in MODEL_COLORS if m in df["model_key"].unique()]
    x = np.arange(len(books))
    width = 0.25
    
    for i, model_key in enumerate(models):
        subset = df[df["model_key"] == model_key]
        means = [subset[subset["book"] == b]["efs"].mean() for b in books]
        
        ax.bar(x + i * width, means, width,
               label=MODEL_LABELS.get(model_key, model_key),
               color=MODEL_COLORS[model_key], alpha=0.85)
    
    book_labels = [b.replace("_", " ").title() for b in books]
    ax.set_xticks(x + width)
    ax.set_xticklabels(book_labels)
    ax.set_ylabel("Mean EFS")
    ax.set_title("Mean EFS by Source Book and Model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    out = FIG_DIR / "fig3_efs_by_book.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {out}")


# ── Plot 4: Correlation Heatmap ─────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame):
    cols = ["delta_f1", "delta_f2", "delta_f3", "delta_f4", "efs"]
    labels = ["Δ Moral", "Δ Agency", "Δ Lexical", "Δ Causal", "EFS"]
    
    corr = df[cols].corr()
    corr.index = labels
    corr.columns = labels
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Correlation Between EFS Dimensions")
    
    out = FIG_DIR / "fig4_correlation_heatmap.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {out}")


def plot_correlation_heatmap(df: pd.DataFrame):
    cols = ["delta_f1", "delta_f2", "delta_f3", "delta_f4", "efs"]
    labels = ["Δ Moral", "Δ Agency", "Δ Lexical", "Δ Causal", "EFS"]
    
    corr = df[cols].corr()
    corr.index = labels
    corr.columns = labels
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, ax=ax, square=True, linewidths=0.5)
    ax.set_title("Correlation Between EFS Dimensions")
    
    out = FIG_DIR / "fig4_correlation_heatmap.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {out}")


# ── Clustering: Distortion Profiles ─────────────────────────────────────────

def cluster_distortion_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cluster passages by their distortion profile (delta_f1 through delta_f4).
    Returns updated dataframe with 'cluster' column.
    """
    X = df[["delta_f1", "delta_f2", "delta_f3", "delta_f4"]].values
    X_scaled = StandardScaler().fit_transform(X)
    
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)
    
    print(f"\n  === Distortion Profile Clusters ===")
    print(f"  Cluster centers (scaled space):")
    for i, center in enumerate(km.cluster_centers_):
        print(f"    Cluster {i}: {center}")
    
    for i in range(3):
        cluster_subset = df[df["cluster"] == i]
        print(f"  Cluster {i}: {len(cluster_subset)} passages (mean EFS={cluster_subset['efs'].mean():.4f})")
    
    return df


def plot_clusters(df: pd.DataFrame):
    """Scatter plot of clusters by first two dimensions."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for cluster_id in sorted(df["cluster"].unique()):
        subset = df[df["cluster"] == cluster_id]
        ax.scatter(subset["delta_f1"], subset["delta_f2"],
                   label=f"Cluster {cluster_id}", alpha=0.6, s=50)
    
    ax.set_xlabel("Δ Moral Explicitness")
    ax.set_ylabel("Δ Agency Attribution")
    ax.set_title("Distortion Profile Clusters")
    ax.legend()
    ax.grid(alpha=0.3)
    
    out = FIG_DIR / "fig5_clusters.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {out}")


# ── Anomaly Detection ───────────────────────────────────────────────────────

def flag_anomalies(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    """
    Flag anomalies as EFS values beyond ±threshold standard deviations.
    Returns updated dataframe with 'is_anomaly' column.
    """
    mu, sigma = df["efs"].mean(), df["efs"].std()
    df["is_anomaly"] = (df["efs"] - mu).abs() > threshold * sigma
    
    n_anomalies = df["is_anomaly"].sum()
    print(f"\n  === Anomaly Detection (threshold = {threshold}σ) ===")
    print(f"  Found {n_anomalies} anomalies out of {len(df)} passages ({100*n_anomalies/len(df):.1f}%)")
    
    if n_anomalies > 0:
        print(f"\n  Top anomalous passages (by EFS magnitude):")
        anomaly_subset = df[df["is_anomaly"]].nlargest(5, "efs")[["book", "model_key", "efs", "delta_f1", "delta_f2"]]
        print(anomaly_subset.to_string())
    
    return df


def plot_anomalies(df: pd.DataFrame):
    """Highlight anomalies in the EFS distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    normal = df[~df["is_anomaly"]]
    anomalies = df[df["is_anomaly"]]
    
    ax.hist(normal["efs"], bins=30, alpha=0.7, color="blue", label="Normal")
    ax.hist(anomalies["efs"], bins=10, alpha=0.9, color="red", label="Anomalies")
    
    ax.set_xlabel("Epistemic Fidelity Score (EFS)")
    ax.set_ylabel("Frequency")
    ax.set_title("EFS Distribution with Detected Anomalies")
    ax.legend()
    ax.grid(alpha=0.3)
    
    out = FIG_DIR / "fig6_anomalies.pdf"
    plt.savefig(out)
    plt.savefig(str(out).replace(".pdf", ".png"))
    plt.close()
    print(f"  Saved {out}")


# ── Ablation Analysis: Feature Sensitivity ──────────────────────────────────

def ablation_analysis(df: pd.DataFrame, weights: np.ndarray) -> dict:
    """
    Perform ablation analysis: remove each dimension and observe EFS shift.
    Returns dict with results for each ablation.
    """
    FEATURES = ["delta_f1", "delta_f2", "delta_f3", "delta_f4"]
    feature_names = ["Moral Expl.", "Agency Attr.", "Lexical Int.", "Causal Compl."]
    results = {}
    
    baseline_efs = (df[FEATURES].values @ weights).mean()
    
    print(f"\n  === Ablation Analysis: Feature Importance ===")
    print(f"  Baseline EFS (all features): {baseline_efs:.4f}")
    print(f"  Feature ablations:")
    
    for drop_idx, (feat, fname) in enumerate(zip(FEATURES, feature_names)):
        w_ablated = weights.copy()
        w_ablated[drop_idx] = 0.0
        efs_ablated = (df[FEATURES].values @ w_ablated).mean()
        efs_change = baseline_efs - efs_ablated
        results[feat] = efs_change
        
        print(f"    Drop {fname:<15}: EFS change = {efs_change:+.4f} "
              f"(weight importance: {efs_change/baseline_efs*100:+.1f}%)")
    
    return results


# ── Statistics Table ────────────────────────────────────────────────────────

def print_stats_table(df: pd.DataFrame):
    print("\n  === Full Statistics Table (for paper) ===")
    
    dims = ["delta_f1", "delta_f2", "delta_f3", "delta_f4", "efs"]
    dim_labels = ["Δ Moral Expl.", "Δ Agency Attr.",
                  "Δ Lexical Int.", "Δ Causal Compl.", "EFS"]
    
    for model_key in sorted(df["model_key"].unique()):
        subset = df[df["model_key"] == model_key]
        print(f"\n  {MODEL_LABELS.get(model_key, model_key)} (n={len(subset)})")
        print(f"  {'Dimension':<20} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"  {'-'*52}")
        for dim, label in zip(dims, dim_labels):
            col = subset[dim]
            print(f"  {label:<20} {col.mean():>8.4f} {col.std():>8.4f} "
                  f"{col.min():>8.4f} {col.max():>8.4f}")
    
    # Statistical significance between models
    models = sorted(df["model_key"].unique())
    if len(models) >= 2:
        print("\n  === Pairwise t-tests on EFS ===")
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                a = df[df["model_key"] == models[i]]["efs"]
                b = df[df["model_key"] == models[j]]["efs"]
                t_stat, p_val = stats.ttest_ind(a, b)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"  {models[i]} vs {models[j]}: t={t_stat:.3f}, p={p_val:.4f} {sig}")
    
    # Save as CSV for paper
    out_file = RESULTS_DIR / "summary_statistics.csv"
    summary = df.groupby("model_key")[dims].agg(["mean","std","min","max"])
    summary.to_csv(out_file)
    print(f"\n  Saved summary statistics to {out_file}")


if __name__ == "__main__":
    print("=== Step 5: Generating analysis and figures ===\n")
    df = load_data()
    print(f"  Loaded {len(df)} rows across {df['model_key'].nunique()} models.\n")
    
    # Add distortion direction column
    df["distortion_direction"] = np.sign(df["efs"])
    n_enriched = (df["distortion_direction"] < 0).sum()
    n_attenuated = (df["distortion_direction"] > 0).sum()
    print(f"  Distortion direction: {n_enriched} enriched, {n_attenuated} attenuated")
    print(f"  (Negative EFS = model enriched moral content)\n")
    
    # Generate standard plots
    plot_efs_distribution(df)
    plot_dimension_comparison(df)
    plot_efs_by_book(df)
    plot_correlation_heatmap(df)
    
    # Clustering and anomaly detection (Section IV-D)
    df = cluster_distortion_profiles(df)
    plot_clusters(df)
    
    df = flag_anomalies(df, threshold=2.0)
    plot_anomalies(df)
    
    # Ablation analysis (Section V-E)
    # Extract weights from EFS computation (default or learned)
    import numpy as np
    from config import DEFAULT_WEIGHTS
    # If learned weights exist, use those; otherwise use defaults
    weights = np.array(DEFAULT_WEIGHTS)
    ablation_results = ablation_analysis(df, weights)
    
    # Statistics and paper-ready output
    print_stats_table(df)
    
    print(f"\nAll figures saved to {FIG_DIR}")
    print("You're ready to put results in your paper!")