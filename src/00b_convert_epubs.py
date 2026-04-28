"""
Convert EPUB files to plain text for books not on Gutenberg.
Place your .epub files in data/raw/ and run this once.

Usage:
    python src/00b_convert_epubs.py

This script:
1. Finds all .epub files in data/raw/
2. Extracts plain text (strips HTML, formatting, navigation)
3. Saves as .txt files alongside the EPUBs
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
except ImportError:
    print("❌ Missing required packages: ebooklib, beautifulsoup4")
    print("   Install with: pip install ebooklib beautifulsoup4")
    sys.exit(1)

from config import RAW_DIR


def epub_to_text(epub_path: Path) -> str:
    """
    Extract plain text from an EPUB file.
    Skips navigation/ToC pages and strips HTML formatting.
    """
    book = epub.read_epub(str(epub_path))
    chapters = []
    
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), "html.parser")
            
            # Try to extract just the body content (skip nav/headers)
            body = soup.find("body")
            if body:
                text = body.get_text(separator=" ", strip=True)
            else:
                text = soup.get_text(separator=" ", strip=True)
            
            # Skip pages with too little text (nav, toc, blank pages)
            if len(text.split()) > 50:
                chapters.append(text)
    
    return "\n\n".join(chapters)


if __name__ == "__main__":
    print("=== Converting EPUB files to plain text ===\n")
    
    epub_files = list(RAW_DIR.glob("*.epub"))
    
    if not epub_files:
        print("  ℹ️  No .epub files found in data/raw/")
        print("     Place your EPUB files there and run again.")
        sys.exit(0)
    
    for epub_file in epub_files:
        out_path = epub_file.with_suffix(".txt")
        
        if out_path.exists():
            print(f"  [{epub_file.stem}] Already exists, skipping.")
            continue
        
        print(f"  [{epub_file.stem}] Converting {epub_file.name}...")
        
        try:
            text = epub_to_text(epub_file)
            word_count = len(text.split())
            out_path.write_text(text, encoding="utf-8")
            print(f"    ✓ Saved {word_count:,} words to {out_path.name}\n")
        except Exception as e:
            print(f"    ✗ Error: {e}\n")
    
    print("Done. Check data/raw/ for .txt files.")
    print("Then run: python src/00_download_books.py")
