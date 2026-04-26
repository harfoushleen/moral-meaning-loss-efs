"""
STEP 3: Extract the 4 EFS feature dimensions from every (S, I) pair.
Outputs data/results/features_all.csv

Run: python src/03_extract_features.py
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nrclex import NRCLex
from tqdm import tqdm

from config import RESULTS_DIR

nlp   = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()

# Causal connectives for f4
CAUSAL_CONNECTIVES = {
    "because", "therefore", "thus", "hence", "consequently",
    "since", "so", "accordingly", "as a result", "due to",
    "owing to", "for this reason", "thereby", "leads to"
}

# Moral/emotional NRC categories
MORAL_EMOTIONS = {"anger", "fear", "disgust", "sadness", "trust"}


def extract_features(text: str) -> dict:
    """
    Extract all 4 EFS dimensions from a text.
    Returns dict with keys f1, f2, f3, f4.
    """
    doc = spacy.tokens.Doc  # just for type hint
    doc = nlp(text)
    tokens_lower = [t.text.lower() for t in doc]
    n_tokens = len(tokens_lower) + 1e-9
    sentences = list(doc.sents)
    n_sents = len(sentences) + 1e-9

    # ── f1: Moral Explicitness ─────────────────────────────────────────────
    # VADER compound (abs value = emotional intensity regardless of polarity)
    vader_compound = abs(vader.polarity_scores(text)["compound"])
    
    # NRC moral emotion frequency
    try:
        nrc = NRCLex(text)
        freqs = nrc.affect_frequencies
        nrc_moral = sum(freqs.get(e, 0.0) for e in MORAL_EMOTIONS)
    except Exception:
        nrc_moral = 0.0
    
    f1 = (vader_compound + nrc_moral) / 2.0

    # ── f2: Agency Attribution ─────────────────────────────────────────────
    # Active agent constructions (nsubj of a VERB) vs passive (nsubjpass)
    active_agents = sum(
        1 for token in doc
        if token.dep_ == "nsubj" and token.head.pos_ == "VERB"
    )
    passive_count = sum(
        1 for token in doc
        if token.dep_ in ("nsubjpass", "auxpass")
    )
    # Agency = ratio of active to total subject constructions
    total_subj = active_agents + passive_count + 1e-9
    f2 = active_agents / total_subj

    # ── f3: Lexical Intensity ──────────────────────────────────────────────
    # Mean absolute VADER score per sentence
    sent_scores = [
        abs(vader.polarity_scores(sent.text)["compound"])
        for sent in sentences
    ]
    f3 = sum(sent_scores) / n_sents

    # ── f4: Causal Completeness ────────────────────────────────────────────
    # Density of causal connective words/phrases
    text_lower = text.lower()
    causal_hits = sum(1 for c in CAUSAL_CONNECTIVES if c in text_lower)
    f4 = causal_hits / n_tokens * 100  # per 100 tokens

    return {"f1": round(f1, 6), "f2": round(f2, 6),
            "f3": round(f3, 6), "f4": round(f4, 6)}


def process_file(interp_file: Path) -> pd.DataFrame:
    df = pd.read_csv(interp_file)
    rows = []
    
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc=f"  Features [{row['model_key'] if 'model_key' in df.columns else interp_file.stem}]"):
        try:
            fs = extract_features(str(row["source_passage"]))
            fi = extract_features(str(row["interpretation"]))
            
            rows.append({
                "passage_id":     row["passage_id"],
                "book":           row["book"],
                "model_key":      row["model_key"],
                "model_name":     row["model_name"],
                # Source features
                "s_f1": fs["f1"], "s_f2": fs["f2"],
                "s_f3": fs["f3"], "s_f4": fs["f4"],
                # Interpretation features
                "i_f1": fi["f1"], "i_f2": fi["f2"],
                "i_f3": fi["f3"], "i_f4": fi["f4"],
                # Deltas (positive = attenuation = distortion)
                "delta_f1": round(fs["f1"] - fi["f1"], 6),
                "delta_f2": round(fs["f2"] - fi["f2"], 6),
                "delta_f3": round(fs["f3"] - fi["f3"], 6),
                "delta_f4": round(fs["f4"] - fi["f4"], 6),
            })
        except Exception as e:
            print(f"\n  ⚠️  Skipping {row['passage_id']}: {e}")
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("=== Step 3: Extracting features ===")
    
    all_dfs = []
    for interp_file in sorted(RESULTS_DIR.glob("interpretations_*.csv")):
        print(f"\nProcessing {interp_file.name}...")
        df = process_file(interp_file)
        all_dfs.append(df)
    
    if not all_dfs:
        print("No interpretation files found. Run 02_generate_interpretations.py first.")
    else:
        combined = pd.concat(all_dfs, ignore_index=True)
        out_file = RESULTS_DIR / "features_all.csv"
        combined.to_csv(out_file, index=False)
        print(f"\nSaved {len(combined)} rows to {out_file}")
        print(combined[["model_key","delta_f1","delta_f2","delta_f3","delta_f4"]].groupby("model_key").mean().round(4))