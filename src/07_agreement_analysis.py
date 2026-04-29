"""
STEP 7: Analyze human annotation agreement and build consensus labels.

Expected input files:
  annotations/annotator_*.csv

Each annotator file should come from annotation_template.csv and contain:
  passage_id, model_key, distortion_score

Outputs:
  annotations/agreement_pairwise.csv
  annotations/agreement_summary.csv
  annotations/annotations_consensus.csv

Optional:
  python src/07_agreement_analysis.py --write-consensus

This also writes annotations/annotations.csv, which 04_compute_efs.py uses for
ridge-regression weight learning.
"""

from pathlib import Path
import argparse
import sys
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from config import ANNOT_DIR

KEYS = ["passage_id", "model_key"]
SCORE = "distortion_score"


def annotator_name(path: Path) -> str:
    return path.stem.replace("annotator_", "")


def load_annotator_files() -> dict[str, pd.DataFrame]:
    files = sorted(ANNOT_DIR.glob("annotator_*.csv"))
    if not files:
        raise FileNotFoundError(
            "No annotator files found. Copy annotation_template.csv to files "
            "like annotations/annotator_perla.csv and fill distortion_score."
        )

    frames = {}
    for path in files:
        df = pd.read_csv(path)
        missing = [col for col in KEYS + [SCORE] if col not in df.columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")

        df = df[KEYS + [SCORE]].copy()
        df[SCORE] = pd.to_numeric(df[SCORE], errors="coerce")
        df = df.dropna(subset=[SCORE])
        df = df[df[SCORE].between(1, 5)]
        df[SCORE] = df[SCORE].round().astype(int)
        frames[annotator_name(path)] = df

    return frames


def pairwise_agreement(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    names = sorted(frames)

    for i, left_name in enumerate(names):
        for right_name in names[i + 1:]:
            left = frames[left_name].rename(columns={SCORE: "left_score"})
            right = frames[right_name].rename(columns={SCORE: "right_score"})
            merged = left.merge(right, on=KEYS, how="inner")

            if merged.empty:
                rows.append({
                    "annotator_a": left_name,
                    "annotator_b": right_name,
                    "shared_rows": 0,
                    "exact_agreement": np.nan,
                    "within_one_agreement": np.nan,
                    "mean_absolute_disagreement": np.nan,
                    "quadratic_weighted_kappa": np.nan,
                })
                continue

            diff = (merged["left_score"] - merged["right_score"]).abs()
            rows.append({
                "annotator_a": left_name,
                "annotator_b": right_name,
                "shared_rows": len(merged),
                "exact_agreement": (diff == 0).mean(),
                "within_one_agreement": (diff <= 1).mean(),
                "mean_absolute_disagreement": diff.mean(),
                "quadratic_weighted_kappa": cohen_kappa_score(
                    merged["left_score"],
                    merged["right_score"],
                    weights="quadratic",
                    labels=[1, 2, 3, 4, 5],
                ),
            })

    columns = [
        "annotator_a",
        "annotator_b",
        "shared_rows",
        "exact_agreement",
        "within_one_agreement",
        "mean_absolute_disagreement",
        "quadratic_weighted_kappa",
    ]
    return pd.DataFrame(rows, columns=columns)


def consensus_annotations(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    stacked = []
    for name, df in frames.items():
        temp = df.copy()
        temp["annotator"] = name
        stacked.append(temp)

    all_scores = pd.concat(stacked, ignore_index=True)
    consensus = (
        all_scores
        .groupby(KEYS)[SCORE]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "distortion_score",
            "std": "score_std",
            "count": "n_annotators",
        })
    )
    consensus["score_std"] = consensus["score_std"].fillna(0.0)
    return consensus


def write_outputs(pairwise: pd.DataFrame, consensus: pd.DataFrame, write_consensus: bool):
    pairwise_path = ANNOT_DIR / "agreement_pairwise.csv"
    summary_path = ANNOT_DIR / "agreement_summary.csv"
    consensus_path = ANNOT_DIR / "annotations_consensus.csv"

    pairwise.to_csv(pairwise_path, index=False)
    consensus.to_csv(consensus_path, index=False)

    annotator_count = len(set(pairwise["annotator_a"]).union(pairwise["annotator_b"])) if not pairwise.empty else 1
    summary = pd.DataFrame([{
        "annotator_files": annotator_count,
        "pairwise_comparisons": len(pairwise),
        "mean_exact_agreement": pairwise["exact_agreement"].mean(),
        "mean_within_one_agreement": pairwise["within_one_agreement"].mean(),
        "mean_absolute_disagreement": pairwise["mean_absolute_disagreement"].mean(),
        "mean_quadratic_weighted_kappa": pairwise["quadratic_weighted_kappa"].mean(),
        "consensus_rows": len(consensus),
    }])
    summary.to_csv(summary_path, index=False)

    if write_consensus:
        final_path = ANNOT_DIR / "annotations.csv"
        consensus.to_csv(final_path, index=False)
        print(f"  Wrote consensus labels for EFS learning: {final_path}")

    print(f"  Wrote pairwise agreement: {pairwise_path}")
    print(f"  Wrote agreement summary: {summary_path}")
    print(f"  Wrote consensus annotations: {consensus_path}")
    print("\n  Agreement summary")
    print(summary.round(4))


def main():
    parser = argparse.ArgumentParser(description="Analyze annotation agreement.")
    parser.add_argument(
        "--write-consensus",
        action="store_true",
        help="Also write annotations/annotations.csv for 04_compute_efs.py.",
    )
    args = parser.parse_args()

    print("=== Step 7: Human annotation agreement ===")
    frames = load_annotator_files()
    print(f"  Loaded {len(frames)} annotator files.")

    pairwise = pairwise_agreement(frames)
    consensus = consensus_annotations(frames)
    write_outputs(pairwise, consensus, args.write_consensus)


if __name__ == "__main__":
    main()
