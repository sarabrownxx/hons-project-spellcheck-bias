"""
scripts/corrections_languagetool.py

Computes LanguageTool spell-checker correction suggestions for names, adding LT
columns to the enriched names parquet.

Modes (--mode):
  chunk   Check one chunk of words via LT and save results to a JSON file.
          Designed to run in parallel (e.g. via GitHub Actions matrix).
          Use with --chunk N --total-chunks M.
  merge   Load the base parquet and all chunk JSON files, assemble LT columns,
          and save data/lt_results.parquet. Does not call the LT server.
  both    Run chunk + merge in a single process (default; for local use).

New columns added:
  lt_orig_known                 — bool, Condition A (original name)
  lt_latin_known                — bool, Condition B (latinised name)
  lt_orig_correction            — str | None, top suggestion for unknowns
  lt_latin_correction           — str | None, top suggestion for unknowns
  lt_orig_correction_in_dataset — bool, whether correction appears in name corpus
  lt_latin_correction_in_dataset— bool, whether correction appears in name corpus

Notes:
  - LanguageTool is case-sensitive. Names are passed to LT as-is (not lowercased),
    unlike hunspell/pysc which lowercase before lookup. This means "Fatima" and
    "fatima" would be checked separately if both appeared — but since all names
    are unique in the dataset, each name is checked exactly once.
  - Ideographic-script rows (CJK, Hangul, Hiragana, Katakana) are excluded from
    Condition A words sent to LT; lt_orig_correction is set to None for these rows.
    Condition B (name_latin) has no ideographic exclusion as anyascii output is Latin.
  - Threading: LanguageTool's Python wrapper is not thread-safe. This script uses
    direct HTTP requests via per-thread requests.Session objects instead.
  - Java 17+ is required to run the LanguageTool server (language_tool_python
    downloads and starts it automatically).

Usage:
  python scripts/corrections_languagetool.py --mode both
  python scripts/corrections_languagetool.py --mode chunk --chunk 0 --total-chunks 4
  python scripts/corrections_languagetool.py --mode merge --total-chunks 4
"""

import argparse
import importlib.metadata
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from math import ceil
from pathlib import Path

import requests as _requests
from dotenv import load_dotenv
load_dotenv()

import pandas as pd

LOGS_DIR = Path("logs")
OUTPUT_PATH = Path("data/final_results.parquet")
IDEOGRAPHIC_SCRIPTS = {"CJK", "Hangul", "Hiragana", "Katakana"}

_thread_local = threading.local()


def _get_session() -> _requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = _requests.Session()
    return _thread_local.session


def _check_one(word: str, lt_url: str) -> tuple:
    """Check a single word with LanguageTool via HTTP.

    Returns (word, known: bool, correction: str | None).
    Treats the word as known on network/parse errors.
    """
    try:
        resp = _get_session().get(
            lt_url, params={"language": "en-US", "text": word}, timeout=30
        )
        resp.raise_for_status()
        matches = [
            m for m in resp.json()["matches"]
            if m["rule"]["issueType"] == "misspelling"
        ]
        known = len(matches) == 0
        correction = (
            matches[0]["replacements"][0]["value"]
            if matches and matches[0]["replacements"]
            else None
        )
        return word, known, correction
    except Exception as e:
        log.warning("LT check failed for %r: %s", word, e)
        return word, True, None  # treat as known on error


def _lt_check_batch(words: list, lt_url: str, n_threads: int, label: str) -> dict:
    """Check a list of words with LanguageTool using a thread pool.

    Returns a dict: {word: (known: bool, correction: str | None)}.
    Logs progress every 5000 words.
    """
    results = {}
    total = len(words)
    t0 = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=n_threads) as ex:
        futures = {ex.submit(_check_one, w, lt_url): w for w in words}
        for fut in as_completed(futures):
            word, known, correction = fut.result()
            results[word] = (known, correction)
            done += 1
            if done % 5_000 == 0:
                log.info("    %s: %s / %s (%.0f%%)  %.0fs elapsed",
                         label, f"{done:,}", f"{total:,}",
                         100 * done / total, time.time() - t0)

    log.info("    %s done — %s words  (%.0fs)",
             label, f"{total:,}", time.time() - t0)
    return results


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


def _chunk_path(chunk: int) -> Path:
    return Path(f"data/lt_chunk_{chunk}.json")


def setup_logging(timestamp: str, mode: str) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"corrections_languagetool_{mode}_{timestamp}.log"
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


def run_chunk(df: pd.DataFrame, chunk: int, total_chunks: int,
              n_threads: int, lt_url: str) -> None:
    """Check one chunk of words and save results to a JSON file."""
    ideographic_mask = df["name_script"].isin(IDEOGRAPHIC_SCRIPTS)

    cond_a_words = list(
        df.loc[~ideographic_mask, "name"].dropna().unique()
    )
    cond_b_words = list(df["name_latin"].dropna().unique())

    all_words = sorted(set(cond_a_words) | set(cond_b_words))
    log.info("  %s total unique words (combined A+B, ideographic excluded from A)",
             f"{len(all_words):,}")

    chunk_size = ceil(len(all_words) / total_chunks)
    start = chunk * chunk_size
    words_this_chunk = all_words[start:start + chunk_size]
    log.info("  Chunk %d: words %s–%s (%s words)",
             chunk, f"{start:,}", f"{start + len(words_this_chunk):,}",
             f"{len(words_this_chunk):,}")

    batch_results = _lt_check_batch(
        words_this_chunk, lt_url, n_threads,
        label=f"lt chunk {chunk}/{total_chunks}"
    )

    known_map = {w: batch_results[w][0] for w in batch_results}
    correction_map = {
        w: batch_results[w][1]
        for w in batch_results
        if not batch_results[w][0]
    }

    out = _chunk_path(chunk)
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"known_map": known_map, "correction_map": correction_map},
                  f, ensure_ascii=False)
    log.info("  Saved chunk %d: %s words (%s unknown) → %s",
             chunk, f"{len(known_map):,}", f"{len(correction_map):,}", out)


def run_merge(df: pd.DataFrame, total_chunks: int) -> None:
    """Load all chunk JSONs, assemble LT columns, and save final parquet."""
    log.info("")
    log.info("[loading chunk JSON files]")

    combined_known_map: dict = {}
    combined_correction_map: dict = {}

    for i in range(total_chunks):
        p = _chunk_path(i)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            combined_known_map.update(data.get("known_map", {}))
            combined_correction_map.update(data.get("correction_map", {}))
            log.info("  Loaded chunk %d: %s known entries, %s corrections",
                     i, f"{len(data.get('known_map', {})):,}",
                     f"{len(data.get('correction_map', {})):,}")
        else:
            log.warning("  Chunk %d not found at %s — skipping.", i, p)

    log.info("  Combined: %s known entries, %s corrections",
             f"{len(combined_known_map):,}", f"{len(combined_correction_map):,}")

    dataset_latin_set = set(df["name_latin"].str.lower().dropna())
    ideographic_mask = df["name_script"].isin(IDEOGRAPHIC_SCRIPTS)

    log.info("")
    log.info("[mapping LT results to dataframe]")

    df["lt_orig_known"] = df["name"].map(combined_known_map).fillna(False).astype(bool)
    df["lt_latin_known"] = df["name_latin"].map(combined_known_map).fillna(False).astype(bool)

    df["lt_orig_correction"] = df["name"].map(combined_correction_map)
    df.loc[df["lt_orig_known"], "lt_orig_correction"] = None
    df.loc[ideographic_mask, "lt_orig_correction"] = None

    df["lt_latin_correction"] = df["name_latin"].map(combined_correction_map)
    df.loc[df["lt_latin_known"], "lt_latin_correction"] = None

    df["lt_orig_correction_in_dataset"] = (
        df["lt_orig_correction"].fillna("").str.lower().isin(dataset_latin_set)
    )
    df["lt_latin_correction_in_dataset"] = (
        df["lt_latin_correction"].fillna("").str.lower().isin(dataset_latin_set)
    )

    pct_orig = 100 * df["lt_orig_known"].sum() / len(df)
    pct_latin = 100 * df["lt_latin_known"].sum() / len(df)
    log.info("  Orig  known: %s / %s  (%.1f%%)",
             f"{df['lt_orig_known'].sum():,}", f"{len(df):,}", pct_orig)
    log.info("  Latin known: %s / %s  (%.1f%%)",
             f"{df['lt_latin_known'].sum():,}", f"{len(df):,}", pct_latin)
    log.info("  Orig  corrections: %s", f"{df['lt_orig_correction'].notna().sum():,}")
    log.info("  Latin corrections: %s", f"{df['lt_latin_correction'].notna().sum():,}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    log.info("Saving to %s…", OUTPUT_PATH)
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info("  Saved %s rows, %s columns", f"{len(df):,}", f"{len(df.columns):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["chunk", "merge", "both"], default="both")
    parser.add_argument("--chunk", type=int, default=0,
                        help="0-indexed chunk number (chunk mode)")
    parser.add_argument("--total-chunks", type=int, default=4,
                        help="Total number of chunks")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of threads for HTTP requests (default 8)")
    args = parser.parse_args()
    mode = args.mode

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = setup_logging(timestamp, mode)

    log.info("=" * 70)
    log.info("corrections_languagetool.py  mode=%s  chunk=%s/%s  —  %s",
             mode, args.chunk, args.total_chunks, timestamp)
    log.info("=" * 70)
    log.info("language_tool_python %s", _pkg("language_tool_python"))

    parquet_path = _find_parquet()
    log.info("Loading %s…", parquet_path)
    df = pd.read_parquet(parquet_path)
    log.info("  %s names loaded", f"{len(df):,}")

    required = ["name_script", "name_latin"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error("Missing columns: %s — run spellcheck_names.py first.", missing)
        sys.exit(1)

    if mode in ("chunk", "both"):
        import language_tool_python
        log.info("")
        log.info("[starting LanguageTool server]")
        tool = language_tool_python.LanguageTool("en-US")
        lt_url = tool.url.rstrip("/") + "/check"
        log.info("  LT version: %s", tool.language_tool_download_version)
        log.info("  LT URL: %s", lt_url)

        log.info("")
        log.info("[Condition A+B chunk — chunk %d of %d]", args.chunk, args.total_chunks)
        run_chunk(df, args.chunk, args.total_chunks, args.threads, lt_url)

        tool.close()

        if mode == "chunk":
            log.info("Done.  Log: %s", log_path)
            return

    if mode in ("merge", "both"):
        log.info("")
        log.info("[merge — assembling LT columns from %d chunk(s)]", args.total_chunks)
        run_merge(df, args.total_chunks)

    log.info("Done.  Log: %s", log_path)


if __name__ == "__main__":
    main()
