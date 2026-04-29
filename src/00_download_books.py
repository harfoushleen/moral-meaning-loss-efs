"""
STEP 0: Download public domain books from Project Gutenberg.
Run this ONCE before anything else.

All three books are freely available on Gutenberg — no manual steps needed.
"""

import time
import requests
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from config import RAW_DIR, BOOKS

# Gutenberg serves plain UTF-8 text at this URL pattern.
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

def download_book(book_name: str, gutenberg_id: int):
    out_path = RAW_DIR / BOOKS[book_name]["file"]

    if out_path.exists():
        print(f"  [{book_name}] Already exists, skipping.")
        return

    url = GUTENBERG_URL.format(id=gutenberg_id)
    print(f"  [{book_name}] Downloading from {url} ...")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # Gutenberg sometimes serves latin-1; normalise to UTF-8
    text = response.content.decode("utf-8", errors="replace")
    out_path.write_text(text, encoding="utf-8")
    print(f"  [{book_name}] Saved {len(text):,} chars to {out_path}")
    time.sleep(1)   # be polite to Gutenberg servers


if __name__ == "__main__":
    print("=== Step 0: Downloading books ===")
    for name, meta in BOOKS.items():
        download_book(name, meta["gutenberg_id"])
    print("\nDone. Check data/raw/")
