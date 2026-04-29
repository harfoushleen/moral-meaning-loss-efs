"""
Central configuration for the entire project.
Edit MODELS and BOOKS here to change scope.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

# Always load .env from the project root (one level up from src/)
# This works regardless of which directory you run the script from
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR     = Path(__file__).parent.parent
DATA_DIR     = ROOT_DIR / "data"
RAW_DIR      = DATA_DIR / "raw"
PASSAGES_DIR = DATA_DIR / "passages"
RESULTS_DIR  = DATA_DIR / "results"
ANNOT_DIR    = ROOT_DIR / "annotations"
FIG_DIR      = ROOT_DIR / "outputs" / "figures"

for d in [RAW_DIR, PASSAGES_DIR, RESULTS_DIR, ANNOT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Models ─────────────────────────────────────────────────────────────────
# Actual API model strings used in experiments.
# Paper should cite these exact strings for reproducibility.
MODELS = {
    "gemma": "gemma3:4b",
    "phi":   "phi4-mini",
}

MODEL_DISPLAY_NAMES = {
    "gemma": "Gemma 3 4B",
    "phi":   "Phi-4 Mini",
}

# ── Books ──────────────────────────────────────────────────────────────────
# All three are public domain and freely available on Project Gutenberg.
#
#   We (1924) by Zamyatin — ID 61963
#     The original proto-dystopia: surveillance state, forced confessions,
#     explicit agents of power. Direct inspiration for 1984.
#
#   The Iron Heel (1908) by Jack London — ID 1164
#     Oligarchic oppression, class violence, named perpetrators.
#     Very morally explicit with clear causal chains.
#
#   The Sleeper Awakes (1910) by H.G. Wells — ID 12163
#     Labour tyranny, exploitation, uprising against a ruling class.
#     Strong moral structure throughout.
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


# ── Passage Settings ───────────────────────────────────────────────────────
PASSAGE_MIN_WORDS        = 60
PASSAGE_MAX_WORDS        = 150
TARGET_PASSAGES_PER_BOOK = 50   # 50 x 6 books x 2 models = 600 (S, I) pairs

# ── EFS Settings ──────────────────────────────────────────────────────────
# Default weights used when no annotations exist.
# Order: [f1_moral_explicitness, f2_agency_attribution,
#         f3_lexical_intensity,  f4_causal_completeness]
DEFAULT_WEIGHTS = [0.30, 0.25, 0.25, 0.20]

# Minimum annotated pairs required before ridge regression is trusted.
# Below this threshold the code falls back to default weights with a warning.
MIN_ANNOTATION_PAIRS = 30

# ── Prompt ─────────────────────────────────────────────────────────────────
INTERPRETATION_PROMPT = """You are a literary analyst. Read the following passage from a dystopian novel and provide an interpretation in 3-5 sentences. Explain its meaning, themes, and significance.

Passage:
{passage}

Interpretation:"""
