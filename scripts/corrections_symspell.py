"""
scripts/corrections_symspell.py

Step 3 in the pipeline chain:
  Main pipeline → corrections_hunspell → corrections_symspell → corrections_languagetool

Merges hunspell correction chunk JSONs into the parquet, then adds SymSpell
known/unknown flags and correction suggestions. Output is saved back to
data/names_results_base.parquet so corrections_languagetool.py can find it
via the standard _find_parquet() lookup.

New columns added:
  symspell_orig_known            — bool, Condition A (original name)
  symspell_latin_known           — bool, Condition B (latinised name)
  symspell_orig_correction       — str | None, top SymSpell suggestion
  symspell_latin_correction      — str | None
  symspell_orig_correction_match — dict | None
  symspell_latin_correction_match— dict | None

Notes:
  - SymSpell lookups use lowercased words (frequency dict is all-lowercase).
  - Ideographic-script rows (CJK, Hangul, Hiragana, Katakana) are excluded from
    Condition A corrections; *_orig_correction is None for these rows.

Usage:
  python scripts/corrections_symspell.py [--workers N]
"""

import argparse
import concurrent.futures
import importlib.metadata
import importlib.resources
import logging
import sys
import time
from datetime import datetime
from math import ceil
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from symspellpy import SymSpell, Verbosity
from corrections_utils import build_name_lookup

LOGS_DIR = Path("logs")
OUTPUT_PATH = Path("data/symspell_results.parquet")
IDEOGRAPHIC_SCRIPTS = {"CJK", "Hangul", "Hiragana", "Katakana"}


def _find_parquet() -> Path:
    """Locate the enriched input parquet, tolerating different artifact layouts."""
    candidates = [
        Path("data/names_results_base.parquet"),
        Path("data/names_base.parquet"),
        Path("names_results_base.parquet"),
        Path("names_base.parquet"),
    ]
    for p in candidates:
        if p.exists():
            log.info("Found input parquet at: %s", p)
            return p

    found = sorted(Path(".").rglob("*.parquet"))
    log.info("Parquet search found: %s", [str(f) for f in found])
    if len(found) == 1:
        log.info("Using: %s", found[0])
        return found[0]
    raise FileNotFoundError(
        f"No input parquet found. Searched candidates and workspace. "
        f"All .parquet files found: {[str(f) for f in found]}"
    )


def _load_symspell() -> SymSpell:
    """Load a SymSpell instance with the bundled English unigram frequency dict."""
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    dict_path = importlib.resources.files("symspellpy").joinpath(
        "frequency_dictionary_en_82_765.txt"
    )
    sym.load_dictionary(str(dict_path), term_index=0, count_index=1)
    return sym


def setup_logging(timestamp: str) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"corrections_symspell_{timestamp}.log"
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


def _pkg(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


# Module-level worker function required for ProcessPoolExecutor pickling
def _correct_batch(batch: list) -> list:
    """Worker function: creates its own SymSpell instance and corrects a batch.

    Returns a list of (word, correction_or_None) tuples.
    """
    sym = _load_symspell()
    results = []
    for w in batch:
        if w:
            hits = sym.lookup(w.lower(), Verbosity.TOP,
                              max_edit_distance=2, include_unknown=False)
            correction = hits[0].term if hits else None
            results.append((w, correction))
        else:
            results.append((w, None))
    return results


def symspell_corrections(unknowns: list, sym: SymSpell, workers: int) -> dict:
    """Compute SymSpell corrections for a list of unknown words.

    Returns a dict mapping lowercased word → correction (str or None).
    """
    total = len(unknowns)
    t0 = time.time()

    if workers > 1:
        log.info("  Using %d worker processes", workers)
        chunk_size = ceil(total / workers)
        batches = [unknowns[i:i + chunk_size] for i in range(0, total, chunk_size)]
        result = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_correct_batch, b): b for b in batches}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                batch_results = fut.result()
                for w, correction in batch_results:
                    if w:
                        result[w.lower()] = correction
                done += len(batch_results)
                log.info("    symspell corrections: %s / %s (%.0f%%)  %.0fs elapsed",
                         f"{done:,}", f"{total:,}", 100 * done / total,
                         time.time() - t0)
    else:
        result = {}
        for i, w in enumerate(unknowns):
            if w:
                hits = sym.lookup(w.lower(), Verbosity.TOP,
                                  max_edit_distance=2, include_unknown=False)
                result[w.lower()] = hits[0].term if hits else None
            if (i + 1) % 50_000 == 0:
                log.info("    symspell corrections: %s / %s (%.0f%%)  %.0fs elapsed",
                         f"{i+1:,}", f"{total:,}", 100 * (i + 1) / total,
                         time.time() - t0)

    log.info("    symspell corrections done — %s words  (%.0fs)",
             f"{total:,}", time.time() - t0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of worker processes for corrections (default 1)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = setup_logging(timestamp)

    log.info("=" * 70)
    log.info("corrections_symspell.py  —  %s", timestamp)
    log.info("=" * 70)
    log.info("symspellpy %s", _pkg("symspellpy"))
    log.info("workers=%d", args.workers)

    parquet_path = _find_parquet()
    log.info("Loading %s…", parquet_path)
    df = pd.read_parquet(parquet_path)
    log.info("  %s names loaded", f"{len(df):,}")

    required = ["name_script", "name_latin"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error("Missing columns: %s — run spellcheck_names.py first.", missing)
        sys.exit(1)

    name_lookup = build_name_lookup(df)
    log.info("  Name lookup built: %s entries", f"{len(name_lookup):,}")
    ideographic_mask = df["name_script"].isin(IDEOGRAPHIC_SCRIPTS)

    log.info("")
    log.info("[loading SymSpell frequency dictionary]")
    log.info("")
    log.info("[loading SymSpell frequency dictionary]")
    t0 = time.time()
    sym = _load_symspell()
    log.info("  Loaded %s entries  (%.1fs)", f"{len(sym._words):,}", time.time() - t0)

    # Condition A: known check (original name)
    log.info("")
    log.info("[Condition A — original name]")
    names_a = df["name"].dropna().unique()
    known_a = {w for w in names_a if w and w.lower() in sym._words}
    df["symspell_orig_known"] = df["name"].isin(known_a)
    pct_a = 100 * df["symspell_orig_known"].sum() / len(df)
    log.info("  %s / %s known  (%.1f%%)",
             f"{df['symspell_orig_known'].sum():,}", f"{len(df):,}", pct_a)

    # Condition B: known check (latinised name)
    log.info("")
    log.info("[Condition B — latinised name]")
    names_b = df["name_latin"].dropna().unique()
    known_b = {w for w in names_b if w and w.lower() in sym._words}
    df["symspell_latin_known"] = df["name_latin"].isin(known_b)
    pct_b = 100 * df["symspell_latin_known"].sum() / len(df)
    log.info("  %s / %s known  (%.1f%%)",
             f"{df['symspell_latin_known'].sum():,}", f"{len(df):,}", pct_b)

    # Corrections (combined unknowns, single pass)
    log.info("")
    log.info("[corrections — combined A+B unknowns, ideographic excluded]")

    unknowns_a = set(df.loc[~df["symspell_orig_known"], "name"].dropna().unique())
    unknowns_b = set(df.loc[~df["symspell_latin_known"], "name_latin"].dropna().unique())
    ideographic_words = set(
        df.loc[ideographic_mask, "name"].dropna().unique()
    )
    unknowns_a_lowered = {w.lower() for w in unknowns_a} - {w.lower() for w in ideographic_words}
    unknowns_b_lowered = {w.lower() for w in unknowns_b}

    all_unknowns = sorted(set(unknowns_a_lowered) | set(unknowns_b_lowered))
    log.info("  %s total unique unknowns (combined A+B, ideographic excluded)",
             f"{len(all_unknowns):,}")

    corr_map = symspell_corrections(all_unknowns, sym, args.workers)

    # Map corrections back to df
    log.info("")
    log.info("[mapping corrections to dataframe]")

    df["symspell_orig_correction"] = df["name"].str.lower().map(corr_map)
    df.loc[df["symspell_orig_known"], "symspell_orig_correction"] = None
    df.loc[ideographic_mask, "symspell_orig_correction"] = None

    df["symspell_latin_correction"] = df["name_latin"].str.lower().map(corr_map)
    df.loc[df["symspell_latin_known"], "symspell_latin_correction"] = None

    df["symspell_orig_correction_match"]  = df["symspell_orig_correction"].str.lower().map(name_lookup)
    df["symspell_latin_correction_match"] = df["symspell_latin_correction"].str.lower().map(name_lookup)

    n_orig_corr = df["symspell_orig_correction"].notna().sum()
    n_latin_corr = df["symspell_latin_correction"].notna().sum()
    log.info("  Orig  corrections: %s", f"{n_orig_corr:,}")
    log.info("  Latin corrections: %s", f"{n_latin_corr:,}")

    # Save
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    log.info("Saving to %s…", OUTPUT_PATH)
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info("  Saved %s rows, %s columns", f"{len(df):,}", f"{len(df.columns):,}")

    log.info("Done.  Log: %s", log_path)


if __name__ == "__main__":
    main()
