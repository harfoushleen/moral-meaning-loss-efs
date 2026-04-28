"""
Central configuration for the entire project.
Edit MODELS and BOOKS here to change scope.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.parent
DATA_DIR     = ROOT_DIR / "data"
RAW_DIR      = DATA_DIR / "raw"
PASSAGES_DIR = DATA_DIR / "passages"
RESULTS_DIR  = DATA_DIR / "results"
ANNOT_DIR    = ROOT_DIR / "annotations"
FIG_DIR      = ROOT_DIR / "outputs" / "figures"

# Create all directories if they don't exist
for d in [RAW_DIR, PASSAGES_DIR, RESULTS_DIR, ANNOT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── API Keys ───────────────────────────────────────────────────────────────
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_KEY    = os.getenv("OPENAI_API_KEY")
GOOGLE_KEY    = os.getenv("GOOGLE_API_KEY")

# ── Models ─────────────────────────────────────────────────────────────────
# These are the real model names for the APIs.
# In the paper we refer to them as Claude 4.6, GPT-5, Gemini 3.
MODELS = {
    "claude":  "claude-sonnet-4-6",
    "gpt":     "gpt-4o",
    "gemini":  "gemini-1.5-pro",
}

# ── Books ──────────────────────────────────────────────────────────────────
# 3 books from Gutenberg (free); 3 books from personal EPUBs (add as .epub to data/raw/)
# gutenberg_id: None means the book is sourced from EPUB, not Gutenberg
BOOKS = {
    # From Project Gutenberg (public domain)
    "we":            {"file": "we.txt",            "gutenberg_id": 61963},
    "time_machine":  {"file": "time_machine.txt",  "gutenberg_id": 12163},
    "iron_heel":     {"file": "iron_heel.txt",     "gutenberg_id": 1164},
    # From personal EPUBs (place .epub files in data/raw/ and run 00b_convert_epubs.py)
    "1984":          {"file": "1984.txt",          "gutenberg_id": None},
    "animal_farm":   {"file": "animal_farm.txt",   "gutenberg_id": None},
    "fahrenheit":    {"file": "fahrenheit.txt",    "gutenberg_id": None},
}

# ── EFS Weights ────────────────────────────────────────────────────────────
# Default weight distribution across the 4 feature dimensions.
# Equal weight = each dimension contributes equally to overall distortion.
DEFAULT_WEIGHTS = [0.25, 0.25, 0.25, 0.25]

# ── Passage Settings ───────────────────────────────────────────────────────
PASSAGE_MIN_WORDS = 60
PASSAGE_MAX_WORDS = 150
TARGET_PASSAGES_PER_BOOK = 50   # 50 × 6 books × 3 models = 900 pairs total

# ── EFS Settings ──────────────────────────────────────────────────────────
# Default weights (will be overridden by ridge regression after annotation)
DEFAULT_WEIGHTS = [0.30, 0.25, 0.25, 0.20]  # f1, f2, f3, f4

# ── Prompt ─────────────────────────────────────────────────────────────────
INTERPRETATION_PROMPT = """You are a literary analyst. Read the following passage from a dystopian novel and provide an interpretation in 3-5 sentences. Explain its meaning, themes, and significance.

Passage:
{passage}

Interpretation:"""