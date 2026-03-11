"""
scripts/corrections_names.py

Computes spell-checker correction suggestions for names flagged as unknown
by spellcheck_names.py. Must be run after that script has added the *_known
columns to the parquet.

Modes (--mode):
  hunspell-chunk  Run hunspell on one chunk of unknowns; outputs a JSON
                  correction map. Designed to run in parallel across chunks.
                  Use with --chunk N --total-chunks M.
  pysc            Loads all hunspell chunk JSON files, applies hunspell
                  corrections to the df, then runs pyspellchecker.
                  Use with --total-chunks M.
  both            Run everything sequentially in one process (local use).
"""

import argparse
import importlib.metadata
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from spellchecker import SpellChecker

LOGS_DIR = Path("logs")
OUTPUT_PATH = Path("data/advanced_results_base.parquet")
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


def _chunk_json_path(chunk: int) -> Path:
    return Path(f"data/hunspell_corrections_chunk_{chunk}.json")


def _load_hunspell_chunks(total_chunks: int) -> dict:
    """Merge all hunspell chunk JSON correction maps into one dict."""
    combined = {}
    for i in range(total_chunks):
        p = _chunk_json_path(i)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                combined.update(json.load(f))
            log.info("  Loaded chunk %d: %s corrections (running total %s)",
                     i, f"{len(combined):,}", f"{len(combined):,}")
        else:
            log.warning("  Chunk %d not found at %s — skipping.", i, p)
    return combined


def setup_logging(timestamp, mode):
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / f"corrections_{mode}_{timestamp}.log"
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


def _pkg(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def hunspell_corrections(unknowns, d):
    import enchant
    result = {}
    total = len(unknowns)
    t0 = time.time()
    for i, w in enumerate(unknowns):
        if w:
            try:
                suggestions = d.suggest(w)
                result[w] = suggestions[0] if suggestions else None
            except enchant.errors.Error:
                result[w] = None
        if (i + 1) % 50_000 == 0:
            log.info("    hunspell suggestions: %s / %s (%.0f%%)  %.0fs elapsed",
                     f"{i+1:,}", f"{total:,}", 100 * (i + 1) / total,
                     time.time() - t0)
    log.info("    hunspell suggestions done — %s words  (%.0fs)",
             f"{total:,}", time.time() - t0)
    return result


def pysc_corrections(unknowns, spell):
    result = {}
    total = len(unknowns)
    t0 = time.time()
    for i, w in enumerate(unknowns):
        if w:
            result[w] = spell.correction(w)
        if (i + 1) % 50_000 == 0:
            log.info("    pysc corrections: %s / %s (%.0f%%)  %.0fs elapsed",
                     f"{i+1:,}", f"{total:,}", 100 * (i + 1) / total,
                     time.time() - t0)
    log.info("    pysc corrections done — %s words  (%.0fs)",
             f"{total:,}", time.time() - t0)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hunspell-chunk", "pysc", "both"],
                        default="both")
    parser.add_argument("--chunk", type=int, default=0,
                        help="0-indexed chunk number (hunspell-chunk mode)")
    parser.add_argument("--total-chunks", type=int, default=1,
                        help="Total number of hunspell chunks")
    args = parser.parse_args()
    mode = args.mode

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = setup_logging(timestamp, mode)

    log.info("=" * 70)
    log.info("corrections_names.py  mode=%s  chunk=%s/%s  —  %s",
             mode, args.chunk, args.total_chunks, timestamp)
    log.info("=" * 70)
    log.info("pyspellchecker %s", _pkg("pyspellchecker"))

    parquet_path = _find_parquet()
    log.info("Loading %s…", parquet_path)
    df = pd.read_parquet(parquet_path)
    log.info("  %s names loaded", f"{len(df):,}")

    required = [
        "hunspell_orig_known", "hunspell_latin_known",
        "pysc_orig_known", "pysc_latin_known",
        "name_script", "name_latin",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error("Missing columns: %s — run spellcheck_names.py first.", missing)
        sys.exit(1)

    dataset_latin_set = set(df["name_latin"].str.lower().dropna())
    ideographic_words = set(
        df.loc[df["name_script"].isin(IDEOGRAPHIC_SCRIPTS), "name"].dropna().unique()
    )
    ideographic_mask = df["name_script"].isin(IDEOGRAPHIC_SCRIPTS)

    # ── hunspell-chunk: compute corrections for one chunk of unknowns ──────────
    if mode in ("hunspell-chunk", "both"):
        import enchant
        log.info("pyenchant %s", enchant.__version__)
        log.info("")
        log.info("[hunspell corrections — chunk %d of %d]",
                 args.chunk, args.total_chunks)

        d_hunspell = enchant.Dict("en_US")

        all_unknowns = sorted(
            (set(df.loc[~df["hunspell_orig_known"],  "name"].dropna().unique()) |
             set(df.loc[~df["hunspell_latin_known"], "name_latin"].dropna().unique()))
            - ideographic_words
        )
        log.info("  %s total unique unknowns (combined A+B, ideographic excluded)",
                 f"{len(all_unknowns):,}")

        if args.total_chunks > 1:
            chunk_size = (len(all_unknowns) + args.total_chunks - 1) // args.total_chunks
            start = args.chunk * chunk_size
            chunk_unknowns = all_unknowns[start:start + chunk_size]
            log.info("  Chunk %d: words %s–%s (%s words)",
                     args.chunk, f"{start:,}", f"{start + len(chunk_unknowns):,}",
                     f"{len(chunk_unknowns):,}")
        else:
            chunk_unknowns = all_unknowns

        h_corr_map = hunspell_corrections(chunk_unknowns, d_hunspell)

        if mode == "hunspell-chunk":
            out = _chunk_json_path(args.chunk)
            out.parent.mkdir(exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(h_corr_map, f, ensure_ascii=False)
            log.info("Saved %s corrections to %s", f"{len(h_corr_map):,}", out)
            log.info("Done.  Log: %s", log_path)
            return

    # ── pysc / both: apply hunspell corrections then run pyspellchecker ────────
    if mode in ("pysc", "both"):
        if mode == "pysc":
            log.info("")
            log.info("[loading hunspell corrections from %d chunk(s)]",
                     args.total_chunks)
            h_corr_map = _load_hunspell_chunks(args.total_chunks)
            log.info("  %s total hunspell corrections loaded", f"{len(h_corr_map):,}")

        df["hunspell_orig_correction"]  = df["name"].map(h_corr_map)
        df["hunspell_latin_correction"] = df["name_latin"].map(h_corr_map)
        df.loc[ideographic_mask, "hunspell_orig_correction"] = None
        df["hunspell_orig_correction_in_dataset"]  = df["hunspell_orig_correction"].str.lower().isin(dataset_latin_set)
        df["hunspell_latin_correction_in_dataset"] = df["hunspell_latin_correction"].str.lower().isin(dataset_latin_set)

        log.info("")
        log.info("[pyspellchecker corrections]")
        spell_pysc = SpellChecker()
        p_unknowns = (
            set(df.loc[~df["pysc_orig_known"],  "name"].dropna().unique()) |
            set(df.loc[~df["pysc_latin_known"], "name_latin"].dropna().unique())
        ) - ideographic_words
        log.info("  %s unique unknowns (combined A+B, ideographic excluded)",
                 f"{len(p_unknowns):,}")
        p_corr_map = pysc_corrections(p_unknowns, spell_pysc)

        df["pysc_orig_correction"]  = df["name"].map(p_corr_map)
        df["pysc_latin_correction"] = df["name_latin"].map(p_corr_map)
        df.loc[ideographic_mask, "pysc_orig_correction"] = None
        df["pysc_orig_correction_in_dataset"]  = df["pysc_orig_correction"].str.lower().isin(dataset_latin_set)
        df["pysc_latin_correction_in_dataset"] = df["pysc_latin_correction"].str.lower().isin(dataset_latin_set)

        log.info("Saving to %s…", OUTPUT_PATH)
        df.to_parquet(OUTPUT_PATH, index=False)

    log.info("Done.  Log: %s", log_path)


if __name__ == "__main__":
    main()
