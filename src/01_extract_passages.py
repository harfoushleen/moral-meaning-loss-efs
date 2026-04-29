"""
STEP 1: Extract morally-charged passages from raw book texts.
Outputs one CSV per book in data/passages/

Run: python src/01_extract_passages.py
"""

import re
import csv
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import RAW_DIR, PASSAGES_DIR, BOOKS, PASSAGE_MIN_WORDS, PASSAGE_MAX_WORDS, TARGET_PASSAGES_PER_BOOK

# Words that suggest moral content — used to score/rank passages
MORAL_KEYWORDS = {
    "killed", "murdered", "tortured", "executed", "punished", "oppressed",
    "forced", "commanded", "ordered", "controlled", "power", "authority",
    "obey", "rebel", "resist", "fear", "hate", "rage", "suffer", "betray",
    "destroy", "rule", "dominate", "suppress", "confess", "arrest", "guilty",
    "innocent", "justice", "crime", "violence", "blood", "death", "war",
    "prison", "sentence", "party", "government", "law", "forbidden", "banned"
}

MIN_MORAL_SCORE = 4


def clean_gutenberg_text(text: str) -> str:
    """Remove Gutenberg header/footer boilerplate."""
    start_markers = ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"]
    end_markers   = ["*** END OF THE PROJECT GUTENBERG",   "*** END OF THIS PROJECT GUTENBERG"]

    start_idx = 0
    for marker in start_markers:
        pos = text.upper().find(marker.upper())
        if pos != -1:
            start_idx = text.find("\n", pos) + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        pos = text.upper().find(marker.upper())
        if pos != -1:
            end_idx = pos
            break

    return text[start_idx:end_idx]


def split_into_passages(text: str, min_words: int, max_words: int) -> list:
    """
    Split text into chunks roughly between min_words and max_words.
    Respects sentence boundaries.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if len(s.split()) > 5]

    passages = []
    current_sentences = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_word_count + word_count > max_words and current_word_count >= min_words:
            passages.append(" ".join(current_sentences))
            current_sentences = [sentence]
            current_word_count = word_count
        else:
            current_sentences.append(sentence)
            current_word_count += word_count

    if current_word_count >= min_words:
        passages.append(" ".join(current_sentences))

    return passages


def moral_score(passage: str) -> int:
    """Count how many distinct moral keywords appear in the passage."""
    words = set(passage.lower().split())
    return len(words & MORAL_KEYWORDS)


def extract_and_save(book_name: str):
    raw_file = RAW_DIR / BOOKS[book_name]["file"]
    out_file  = PASSAGES_DIR / f"{book_name}_passages.csv"

    if not raw_file.exists():
        print(f"  [{book_name}] ⚠️  Raw file not found at {raw_file}. Skipping.")
        return

    print(f"  [{book_name}] Reading and cleaning text...")
    text = raw_file.read_text(encoding="utf-8", errors="ignore")
    text = clean_gutenberg_text(text)

    print(f"  [{book_name}] Splitting into passages...")
    all_passages = split_into_passages(text, PASSAGE_MIN_WORDS, PASSAGE_MAX_WORDS)
    print(f"  [{book_name}] Found {len(all_passages)} candidate passages.")

    # Score all passages
    scored = sorted(
        [(moral_score(p), p) for p in all_passages],
        key=lambda x: x[0],
        reverse=True
    )

    # Dynamically lower threshold until enough passages are found
    filtered = []
    threshold = MIN_MORAL_SCORE

    while len(filtered) < TARGET_PASSAGES_PER_BOOK and threshold >= 0:
        filtered = [(score, p) for score, p in scored if score >= threshold]
        if len(filtered) < TARGET_PASSAGES_PER_BOOK:
            print(f"  [{book_name}] Only {len(filtered)} passages at threshold {threshold}, trying {threshold - 1}...")
            threshold -= 1

    top_passages = filtered[:TARGET_PASSAGES_PER_BOOK]
    print(f"  [{book_name}] Final threshold used: {threshold + 1}, selected {len(top_passages)} passages.")

    # Shuffle so they're not all the most extreme
    random.seed(42)
    random.shuffle(top_passages)

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["passage_id", "book", "passage", "moral_keyword_count"])
        for i, (score, passage) in enumerate(top_passages):
            writer.writerow([f"{book_name}_{i:03d}", book_name, passage, score])

    print(f"  [{book_name}] Saved {len(top_passages)} passages to {out_file}")


if __name__ == "__main__":
    print("=== Step 1: Extracting passages ===")
    for book_name in BOOKS:
        extract_and_save(book_name)
    print("\nDone. Check data/passages/")