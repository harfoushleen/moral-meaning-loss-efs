"""
Downloads public domain books from Project Gutenberg.
Run this ONCE before anything else.
Put any books not on Gutenberg (Fahrenheit 451 etc.) manually in data/raw/
"""

import requests
from config import RAW_DIR, BOOKS

GUTENBERG_URLS = [
    "https://www.gutenberg.org/files/{id}/{id}-0.txt",
    "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt",
]

def download_book(book_name: str, gutenberg_id: int):
    out_path = RAW_DIR / BOOKS[book_name]["file"]
    
    if out_path.exists():
        print(f"  [{book_name}] Already exists, skipping.")
        return
    
    # Try each URL format in order
    for url_template in GUTENBERG_URLS:
        url = url_template.format(id=gutenberg_id)
        print(f"  [{book_name}] Trying {url} ...")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            out_path.write_text(response.text, encoding="utf-8")
            print(f"  [{book_name}] ✓ Downloaded and saved to {out_path}")
            return
        except requests.exceptions.RequestException as e:
            print(f"    Failed: {e}")
            continue
    
    print(f"  [{book_name}] ✗ Could not download from any Gutenberg URL.")
    print(f"    Please add {BOOKS[book_name]['file']} manually to data/raw/")


if __name__ == "__main__":
    print("=== Downloading books ===")
    for name, meta in BOOKS.items():
        if meta["gutenberg_id"] is not None:
            download_book(name, meta["gutenberg_id"])
        else:
            path = RAW_DIR / meta["file"]
            if not path.exists():
                print(f"  [{name}] ⚠️  Not on Gutenberg. Please add {meta['file']} to data/raw/ manually.")
            else:
                print(f"  [{name}] Found manually added file.")
    print("\nDone. Check data/raw/ for your files.")