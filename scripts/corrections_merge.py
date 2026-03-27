"""
scripts/corrections_merge.py

Final step in the corrections pipeline. Reads the three tool-specific
parquets (hunspell_results, symspell_results, lt_results) and joins their
correction columns onto the base enriched dataset, producing final_results.parquet.

Each tool parquet contains the full base columns plus its own tool columns.
This script extracts only the tool-specific columns from each and merges them
onto the base by name.

Usage:
  python scripts/corrections_merge.py
  python scripts/corrections_merge.py --lt-language auto
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

LOGS_DIR = Path("logs")

HUNSPELL_PATH  = Path("data/hunspell_results.parquet")
SYMSPELL_PATH  = Path("data/symspell_results.parquet")

HUNSPELL_COLS = [
    "hunspell_orig_correction", "hunspell_latin_correction",
    "hunspell_orig_correction_match", "hunspell_latin_correction_match",
]
SYMSPELL_COLS = [
    "symspell_orig_known", "symspell_latin_known",
    "symspell_orig_correction", "symspell_latin_correction",
    "symspell_orig_correction_match", "symspell_latin_correction_match",
]


def lt_path(lt_language):
    return Path("data/lt_auto_results.parquet") if lt_language == "auto" else Path("data/lt_results.parquet")


def lt_cols(lt_language):
    p = "lt_auto_" if lt_language == "auto" else "lt_"
    return [
        f"{p}orig_known", f"{p}latin_known",
        f"{p}orig_correction", f"{p}latin_correction",
        f"{p}orig_correction_match", f"{p}latin_correction_match",
    ]


def setup_logging(timestamp):
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"corrections_merge_{timestamp}.log"
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(fh)
    root.addHandler(ch)
    return log_path


log = logging.getLogger(__name__)


def _load_tool_cols(path, cols):
    if not path.exists():
        log.warning("  %s not found — skipping", path)
        return None
    df = pd.read_parquet(path, columns=["name"] + [c for c in cols])
    log.info("  Loaded %s  (%s rows)", path, f"{len(df):,}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lt-language", default="en-US",
                        help="'en-US' or 'auto' — selects which LT results file to merge")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = setup_logging(timestamp)

    log.info("=" * 70)
    log.info("corrections_merge.py  lt-language=%s  —  %s", args.lt_language, timestamp)
    log.info("=" * 70)

    output_path = Path("data/final_results.parquet") if args.lt_language == "en-US" \
        else Path("data/final_results_langall.parquet")

    log.info("Loading base from hunspell results…")
    if not HUNSPELL_PATH.exists():
        log.error("%s not found. Run corrections_hunspell workflow first.", HUNSPELL_PATH)
        sys.exit(1)

    base_cols_to_drop = HUNSPELL_COLS + SYMSPELL_COLS + lt_cols(args.lt_language)
    df = pd.read_parquet(HUNSPELL_PATH)
    existing = [c for c in base_cols_to_drop if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    log.info("  Base: %s rows, %s columns", f"{len(df):,}", f"{len(df.columns):,}")

    for path, cols, label in [
        (HUNSPELL_PATH,              HUNSPELL_COLS,          "hunspell"),
        (SYMSPELL_PATH,              SYMSPELL_COLS,          "symspell"),
        (lt_path(args.lt_language),  lt_cols(args.lt_language), "lt"),
    ]:
        tool_df = _load_tool_cols(path, cols)
        if tool_df is None:
            continue
        present = [c for c in cols if c in tool_df.columns]
        df = df.merge(tool_df[["name"] + present], on="name", how="left")
        log.info("  Merged %s columns from %s", len(present), label)

    log.info("Saving to %s…", output_path)
    output_path.parent.mkdir(exist_ok=True)
    df.to_parquet(output_path, index=False)
    log.info("  Saved %s rows, %s columns", f"{len(df):,}", f"{len(df.columns):,}")
    log.info("Done.  Log: %s", log_path)


if __name__ == "__main__":
    main()
