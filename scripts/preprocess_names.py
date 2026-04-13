"""
scripts/preprocess_names.py

Adds Unicode script detection and Latin transliteration columns to
data/names_base.parquet.

Must be run as pipeline step 2, after database_v2.py and before
enrich_names.py and spellcheck_names.py, both of which depend on
the name_latin column.

New columns
───────────
name_script   Unicode script family of the first alphabetic character
              e.g. Latin, Arabic, CJK, Cyrillic, Devanagari, …
name_latin    anyascii transliteration of the name into ASCII/Latin.
              Identical to `name` for names already in Latin script.

Usage
─────
  python scripts/preprocess_names.py
"""

import sys
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from anyascii import anyascii

# Configuration
PARQUET_PATH = Path("data/names_base.parquet")

# Maps the first word of unicodedata.name() to a clean script label.
_SCRIPT_MAP = {
    "LATIN":      "Latin",      "ARABIC":     "Arabic",
    "CYRILLIC":   "Cyrillic",   "CJK":        "CJK",
    "HANGUL":     "Hangul",     "HEBREW":     "Hebrew",
    "BENGALI":    "Bengali",    "GREEK":      "Greek",
    "GEORGIAN":   "Georgian",   "ETHIOPIC":   "Ethiopic",
    "HIRAGANA":   "Hiragana",   "KATAKANA":   "Katakana",
    "DEVANAGARI": "Devanagari", "KHMER":      "Khmer",
    "THAI":       "Thai",       "MYANMAR":    "Myanmar",
    "TAMIL":      "Tamil",      "ARMENIAN":   "Armenian",
    "SINHALA":    "Sinhala",    "THAANA":     "Thaana",
    "GURMUKHI":   "Gurmukhi",  "GUJARATI":   "Gujarati",
    "MALAYALAM":  "Malayalam",  "KANNADA":    "Kannada",
    "TELUGU":     "Telugu",     "ORIYA":      "Oriya",
    "SYRIAC":     "Syriac",
}


def detect_script(name: str) -> str:
    """Return the Unicode script of the first alphabetic character."""
    for ch in name:
        if ch.isalpha():
            raw = unicodedata.name(ch, "").split()[0]
            return _SCRIPT_MAP.get(raw, raw)
    return "Other"


def main():
    if not PARQUET_PATH.exists():
        print(f"ERROR: {PARQUET_PATH} not found. Run database_v2.py first.")
        sys.exit(1)

    print(f"Loading {PARQUET_PATH}…", flush=True)
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  {len(df):,} names loaded", flush=True)

    t0 = time.time()

    print("Detecting Unicode scripts…", flush=True)
    df["name_script"] = df["name"].apply(detect_script)

    print("Building name_latin via anyascii transliteration…", flush=True)
    df["name_latin"] = df["name"].apply(lambda n: anyascii(n).strip())

    elapsed = time.time() - t0

    # Summary
    dist = df["name_script"].value_counts()
    n_already_latin = int((df["name"] == df["name_latin"]).sum())

    print(f"\nCompleted in {elapsed:.0f}s")
    print(f"\nScript distribution:")
    for script, count in dist.items():
        print(f"  {script:<15} {count:>8,}  ({100 * count / len(df):.1f}%)")
    print(f"\nAlready Latin (name == name_latin): "
          f"{n_already_latin:,} ({100 * n_already_latin / len(df):.1f}%)")

    # Save
    print(f"\nSaving to {PARQUET_PATH}…", flush=True)
    df.to_parquet(PARQUET_PATH, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
