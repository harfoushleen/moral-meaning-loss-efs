"""
STEP 4: Compute the Epistemic Fidelity Score (EFS).

Two modes:
  A) Default weights (no annotation needed) — runs immediately.
  B) Learned weights via ridge regression — needs annotations/annotations.csv

Run: python src/04_compute_efs.py

Outputs:
  data/results/efs_scores.csv         — full results with EFS per pair
  annotations/annotation_template.csv — fill this in for mode B
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from config import RESULTS_DIR, ANNOT_DIR, DEFAULT_WEIGHTS

FEATURES = ["delta_f1", "delta_f2", "delta_f3", "delta_f4"]
ANNOTATION_FILE = ANNOT_DIR / "annotations.csv"
TEMPLATE_FILE   = ANNOT_DIR / "annotation_template.csv"


def compute_efs_default(df: pd.DataFrame, weights=DEFAULT_WEIGHTS) -> pd.Series:
    """Compute EFS using fixed default weights."""
    w = np.array(weights)
    X = df[FEATURES].values
    return pd.Series(X @ w, name="efs", index=df.index)


def learn_weights(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Learn EFS weights from human annotations via ridge regression.
    annotations.csv must have columns: passage_id, distortion_score (1-5)
    """
    annot = pd.read_csv(ANNOTATION_FILE)
    merged = df.merge(annot, on="passage_id", how="inner")
    
    if len(merged) < 10:
        print(f"  ⚠️  Only {len(merged)} annotated pairs. Need at least 10.")
        print(f"      Falling back to default weights.")
        return np.array(DEFAULT_WEIGHTS), None
    
    X = merged[FEATURES].values
    y = merged["distortion_score"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=5)
    ridge.fit(X_train, y_train)
    
    y_pred = ridge.predict(X_test)
    r2  = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"  Ridge regression: R²={r2:.3f}, MSE={mse:.4f}, α={ridge.alpha_:.3f}")
    print(f"  Learned weights: {dict(zip(FEATURES, ridge.coef_.round(4)))}")
    
    return ridge.coef_, r2


def create_annotation_template(df: pd.DataFrame):
    """Create a CSV template for manual annotation."""
    if TEMPLATE_FILE.exists():
        return
    
    # Sample 30 diverse passages for annotation
    sample = df.groupby("model_key").apply(
        lambda g: g.sample(min(10, len(g)), random_state=42)
    ).reset_index(drop=True)
    
    template = sample[["passage_id", "book", "model_key"]].copy()
    template["distortion_score"] = ""  # Annotator fills this in (1-5)
    template["notes"] = ""             # Optional comments
    
    template.to_csv(TEMPLATE_FILE, index=False)
    print(f"\n  📋 Annotation template saved to {TEMPLATE_FILE}")
    print(f"     Fill in 'distortion_score' (1=faithful, 5=heavily distorted)")
    print(f"     then save as annotations/annotations.csv")


if __name__ == "__main__":
    print("=== Step 4: Computing EFS ===")
    
    features_file = RESULTS_DIR / "features_all.csv"
    if not features_file.exists():
        print("features_all.csv not found. Run 03_extract_features.py first.")
        sys.exit(1)
    
    df = pd.read_csv(features_file)
    print(f"  Loaded {len(df)} (S, I) pairs.")
    
    # Create annotation template for human labeling
    create_annotation_template(df)
    
    # Choose weight mode
    if ANNOTATION_FILE.exists():
        print("\n  ✅ Annotations found. Learning weights via ridge regression...")
        weights, r2 = learn_weights(df)
        weight_mode = "learned"
    else:
        print("\n  ℹ️  No annotations yet. Using default weights.")
        print(f"     Default: {dict(zip(FEATURES, DEFAULT_WEIGHTS))}")
        weights = np.array(DEFAULT_WEIGHTS)
        weight_mode = "default"
        r2 = None
    
    # Compute EFS for all pairs
    X = df[FEATURES].values
    df["efs"] = X @ weights
    df["weight_mode"] = weight_mode
    
    # Save
    out_file = RESULTS_DIR / "efs_scores.csv"
    df.to_csv(out_file, index=False)
    print(f"\n  Saved EFS scores to {out_file}")
    
    # Quick summary
    print("\n  === EFS Summary by Model ===")
    summary = df.groupby("model_key")["efs"].agg(["mean","std","min","max"])
    print(summary.round(4))
    
    print("\n  === EFS Summary by Book ===")
    book_summary = df.groupby("book")["efs"].agg(["mean","std"])
    print(book_summary.round(4))