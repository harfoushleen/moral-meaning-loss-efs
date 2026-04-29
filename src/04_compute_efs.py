"""
STEP 4: Compute the Epistemic Fidelity Score (EFS).
 
Two modes:
  A) Default weights  — runs immediately, no annotation needed.
  B) Learned weights  — needs annotations/annotations.csv filled in first.
 
Run: python src/04_compute_efs.py
 
Outputs:
  data/results/efs_scores.csv          — full results with EFS per pair
  annotations/annotation_template.csv  — fill this in for mode B
"""
 
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
 
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
 
from config import RESULTS_DIR, ANNOT_DIR, DEFAULT_WEIGHTS, MIN_ANNOTATION_PAIRS
 
FEATURES        = ["delta_f1", "delta_f2", "delta_f3", "delta_f4"]
ANNOTATION_FILE = ANNOT_DIR / "annotations.csv"
TEMPLATE_FILE   = ANNOT_DIR / "annotation_template.csv"
ANNOTATION_SAMPLE_PER_MODEL = 75
 
 
def compute_efs_default(df: pd.DataFrame, weights=DEFAULT_WEIGHTS) -> pd.Series:
    """Compute EFS using fixed default weights."""
    w = np.array(weights)
    X = df[FEATURES].values
    return pd.Series(X @ w, name="efs", index=df.index)
 
 
def learn_weights(df: pd.DataFrame):
    """
    Learn EFS weights from human annotations via ridge regression.
    annotations.csv must have columns: passage_id, distortion_score (1-5)
    Returns (weights array, r2 score or None).
    """
    annot = pd.read_csv(ANNOTATION_FILE)
    merge_keys = ["passage_id", "model_key"] if "model_key" in annot.columns else ["passage_id"]
    merged = df.merge(annot, on=merge_keys, how="inner")
 
    if len(merged) < MIN_ANNOTATION_PAIRS:
        print(
            f"\n  ⚠️  WARNING: Only {len(merged)} annotated pairs found "
            f"(minimum recommended: {MIN_ANNOTATION_PAIRS}).\n"
            f"     Ridge regression on this few examples produces unreliable weights.\n"
            f"     Falling back to default weights. Annotate more pairs for better results."
        )
        return np.array(DEFAULT_WEIGHTS), None
 
    X = merged[FEATURES].values
    y = merged["distortion_score"].values
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    cv_folds = min(5, len(X_train))
    ridge    = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=cv_folds)
    ridge.fit(X_train, y_train)
 
    y_pred = ridge.predict(X_test)
    r2     = r2_score(y_test, y_pred)
    mse    = mean_squared_error(y_test, y_pred)
 
    print(f"  Ridge regression: R²={r2:.3f}, MSE={mse:.4f}, α={ridge.alpha_:.3f}")
    print(f"  Learned weights: {dict(zip(FEATURES, ridge.coef_.round(4)))}")
 
    return ridge.coef_, r2
 
 
def create_annotation_template(df: pd.DataFrame):
    """
    Create a CSV template for manual annotation.
    Includes source_passage and interpretation so annotators can read the texts.
    """
    if TEMPLATE_FILE.exists():
        print(f"  📋 Template already exists at {TEMPLATE_FILE} — not overwriting.")
        return
 
    interp_files = list(RESULTS_DIR.glob("interpretations_*.csv"))
    if not interp_files:
        print("  ⚠️  No interpretation files found — cannot build annotation template.")
        return
 
    interp_df = pd.concat([pd.read_csv(f) for f in interp_files], ignore_index=True)
 
    # FIX: use explicit loop instead of groupby.apply to avoid pandas version
    # issues where the group key column gets dropped in newer pandas (2.x+)
    samples = []
    for model_key, group in df.groupby("model_key"):
        samples.append(group.sample(min(ANNOTATION_SAMPLE_PER_MODEL, len(group)), random_state=42))
    sample = pd.concat(samples, ignore_index=True)
 
    # Merge on both keys so the selected interpretation matches its model.
    merged = sample.merge(
        interp_df[["passage_id", "model_key", "source_passage", "interpretation"]],
        on=["passage_id", "model_key"],
        how="left",
        suffixes=("", "_interp")
    )
    # Clean up any duplicate model_key column from the merge
    if "model_key_interp" in merged.columns:
        merged = merged.drop(columns=["model_key_interp"])
 
    template = merged[[
        "passage_id", "book", "model_key",
        "source_passage", "interpretation"
    ]].copy()
    template["distortion_score"] = ""   # 1=faithful, 5=heavily distorted
    template["notes"]            = ""
 
    template.to_csv(TEMPLATE_FILE, index=False)
    print(f"\n  📋 Annotation template saved to {TEMPLATE_FILE}")
    print(f"     Instructions:")
    print(f"       1. Open the CSV in Excel or Google Sheets")
    print(f"       2. Read 'source_passage' and 'interpretation' for each row")
    print(f"       3. Fill in 'distortion_score' (1 = faithful, 5 = heavily distorted)")
    print(f"       4. Save as annotations/annotations.csv")
    print(f"       5. Re-run this script to use learned weights")
 
 
if __name__ == "__main__":
    print("=== Step 4: Computing EFS ===")
 
    features_file = RESULTS_DIR / "features_all.csv"
    if not features_file.exists():
        print("features_all.csv not found. Run 03_extract_features.py first.")
        sys.exit(1)
 
    df = pd.read_csv(features_file)
    print(f"  Loaded {len(df)} (S, I) pairs.")
 
    create_annotation_template(df)
 
    if ANNOTATION_FILE.exists():
        print("\n  ✅ Annotations found. Learning weights via ridge regression...")
        weights, r2 = learn_weights(df)
        weight_mode = "learned"
    else:
        print("\n  ℹ️  No annotations yet. Using default weights.")
        print(f"     Default: {dict(zip(FEATURES, DEFAULT_WEIGHTS))}")
        weights     = np.array(DEFAULT_WEIGHTS)
        weight_mode = "default"
        r2          = None
 
    X              = df[FEATURES].values
    df["efs"]          = X @ weights
    df["weight_mode"]  = weight_mode
 
    out_file = RESULTS_DIR / "efs_scores.csv"
    df.to_csv(out_file, index=False)
    print(f"\n  Saved EFS scores to {out_file}")
 
    print("\n  === EFS Summary by Model ===")
    print(df.groupby("model_key")["efs"].agg(["mean", "std", "min", "max"]).round(4))
 
    print("\n  === EFS Summary by Book ===")
    print(df.groupby("book")["efs"].agg(["mean", "std"]).round(4))
 
