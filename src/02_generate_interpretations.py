"""
STEP 2: Send each passage to configured Ollama models and save interpretations.
Outputs data/results/interpretations_{model}.csv

Run: python src/02_generate_interpretations.py

Each output CSV records the exact model_name string used so results are
reproducible.

"""

import csv
import time
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from tqdm import tqdm

from config import (
    PASSAGES_DIR, RESULTS_DIR, MODELS,
    INTERPRETATION_PROMPT
)
def call_ollama(passage: str, model: str) -> str:
    import ollama
    response = ollama.chat(model=model, messages=[
        {"role": "user", "content": INTERPRETATION_PROMPT.format(passage=passage)}
    ])
    return response["message"]["content"].strip()

MODEL_CALLERS = {
    "gemma": call_ollama,
    "phi":   call_ollama,
}



# ── Main ───────────────────────────────────────────────────────────────────

def check_ollama_ready():
    """Fail fast if the Ollama server is not available."""
    try:
        import ollama
        available = ollama.list()
    except Exception as exc:
        raise RuntimeError(
            "Ollama is not available. Install/start Ollama and pull the "
            "configured models before running this step."
        ) from exc

    models = available.get("models", []) if isinstance(available, dict) else []
    names = {m.get("name") or m.get("model") for m in models if isinstance(m, dict)}
    missing = [model for model in MODELS.values() if model not in names]
    if missing:
        print(
            "  Warning: these configured models were not reported by Ollama: "
            + ", ".join(missing)
        )


def load_all_passages() -> pd.DataFrame:
    dfs = []
    for csv_file in sorted(PASSAGES_DIR.glob("*_passages.csv")):
        dfs.append(pd.read_csv(csv_file))
    if not dfs:
        raise FileNotFoundError("No passage CSVs found. Run 01_extract_passages.py first.")
    return pd.concat(dfs, ignore_index=True)


def generate_for_model(model_key: str, passages_df: pd.DataFrame):
    model_name = MODELS[model_key]
    out_file   = RESULTS_DIR / f"interpretations_{model_key}.csv"
    caller     = MODEL_CALLERS.get(model_key)
    if caller is None:
        raise KeyError(f"No model caller configured for model key: {model_key}")

    # Resume support: skip already-processed passage IDs
    done_ids = set()
    if out_file.exists():
        existing = pd.read_csv(out_file)
        done_ids = set(existing["passage_id"].tolist())
        print(f"  [{model_key}] Resuming — {len(done_ids)} already done.")

    write_header = not out_file.exists()
    with open(out_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            # model_name is recorded so the exact API string is in the data file
            writer.writerow([
                "passage_id", "book", "model_key", "model_name",
                "source_passage", "interpretation"
            ])

        todo = passages_df[~passages_df["passage_id"].isin(done_ids)]
        print(f"  [{model_key}] Generating {len(todo)} interpretations with {model_name}...")

        for _, row in tqdm(todo.iterrows(), total=len(todo), desc=model_key):
            try:
                interp = caller(row["passage"], model_name)
                writer.writerow([
                    row["passage_id"],
                    row["book"],
                    model_key,
                    model_name,          # exact API model string
                    row["passage"],
                    interp
                ])
                f.flush()
                time.sleep(0.1)
            except Exception as e:
                print(f"\n  ⚠️  Error on {row['passage_id']}: {e}")
                time.sleep(2)

    print(f"  [{model_key}] Done. Saved to {out_file}")

if __name__ == "__main__":
    print("=== Step 2: Generating LLM interpretations ===")
    try:
        check_ollama_ready()
    except RuntimeError as exc:
        print(f"  {exc}")
        sys.exit(1)
    passages_df = load_all_passages()
    print(f"  Loaded {len(passages_df)} total passages.\n")

    for model_key in MODELS:
        generate_for_model(model_key, passages_df)

    print("\nDone. Check data/results/interpretations_*.csv")

