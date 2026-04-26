"""
STEP 2: Send each passage to Claude, GPT, and Gemini and save interpretations.
Outputs data/results/interpretations_{model}.csv

Run: python src/02_generate_interpretations.py

NOTE: This costs API credits. With ~150 passages × 3 models = 450 calls.
      At ~300 tokens each, estimated cost is very small (< $2 total).
"""

import csv
import time
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from tqdm import tqdm

from config import (
    PASSAGES_DIR, RESULTS_DIR, MODELS,
    ANTHROPIC_KEY, OPENAI_KEY, GOOGLE_KEY,
    INTERPRETATION_PROMPT
)


# ── API Callers ────────────────────────────────────────────────────────────

def call_claude(passage: str, model: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
    response = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": INTERPRETATION_PROMPT.format(passage=passage)
        }]
    )
    return response.content[0].text.strip()


def call_gpt(passage: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)
    response = client.chat.completions.create(
        model=model,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": INTERPRETATION_PROMPT.format(passage=passage)
        }]
    )
    return response.choices[0].message.content.strip()


def call_gemini(passage: str, model: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_KEY)
    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(
        INTERPRETATION_PROMPT.format(passage=passage)
    )
    return response.text.strip()


MODEL_CALLERS = {
    "claude": call_claude,
    "gpt":    call_gpt,
    "gemini": call_gemini,
}


# ── Main ───────────────────────────────────────────────────────────────────

def load_all_passages() -> pd.DataFrame:
    """Load and combine all passage CSVs."""
    dfs = []
    for csv_file in PASSAGES_DIR.glob("*_passages.csv"):
        dfs.append(pd.read_csv(csv_file))
    if not dfs:
        raise FileNotFoundError("No passage CSVs found. Run 01_extract_passages.py first.")
    return pd.concat(dfs, ignore_index=True)


def generate_for_model(model_key: str, passages_df: pd.DataFrame):
    """Generate interpretations for one model, with resume support."""
    model_name  = MODELS[model_key]
    out_file    = RESULTS_DIR / f"interpretations_{model_key}.csv"
    caller      = MODEL_CALLERS[model_key]
    
    # Resume support: load already-done IDs
    done_ids = set()
    if out_file.exists():
        existing = pd.read_csv(out_file)
        done_ids = set(existing["passage_id"].tolist())
        print(f"  [{model_key}] Resuming — {len(done_ids)} already done.")
    
    # Open CSV in append mode
    write_header = not out_file.exists()
    with open(out_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["passage_id", "book", "model_key", "model_name",
                             "source_passage", "interpretation"])
        
        todo = passages_df[~passages_df["passage_id"].isin(done_ids)]
        print(f"  [{model_key}] Generating {len(todo)} interpretations with {model_name}...")
        
        for _, row in tqdm(todo.iterrows(), total=len(todo), desc=model_key):
            try:
                interpretation = caller(row["passage"], model_name)
                writer.writerow([
                    row["passage_id"],
                    row["book"],
                    model_key,
                    model_name,
                    row["passage"],
                    interpretation
                ])
                f.flush()  # Write immediately in case of crash
                time.sleep(0.5)  # Be polite to APIs
                
            except Exception as e:
                print(f"\n  ⚠️  Error on {row['passage_id']}: {e}")
                time.sleep(2)  # Wait longer after error
    
    print(f"  [{model_key}] Done. Saved to {out_file}")


if __name__ == "__main__":
    print("=== Step 2: Generating LLM interpretations ===")
    passages_df = load_all_passages()
    print(f"  Loaded {len(passages_df)} total passages.\n")
    
    for model_key in MODELS:
        # Skip if API key not set
        if model_key == "claude"  and not ANTHROPIC_KEY:
            print(f"  [{model_key}] Skipping — ANTHROPIC_API_KEY not set.")
            continue
        if model_key == "gpt"     and not OPENAI_KEY:
            print(f"  [{model_key}] Skipping — OPENAI_API_KEY not set.")
            continue
        if model_key == "gemini"  and not GOOGLE_KEY:
            print(f"  [{model_key}] Skipping — GOOGLE_API_KEY not set.")
            continue
        
        generate_for_model(model_key, passages_df)
    
    print("\nDone. Check data/results/interpretations_*.csv")