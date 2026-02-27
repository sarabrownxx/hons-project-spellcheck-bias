"""
scripts/corrections_names.py

Computes spell-checker correction suggestions for names flagged as unknown
by spellcheck_names.py. Must be run after that script has added the *_known
columns to the parquet.

Adds eight columns: hunspell_orig_correction, hunspell_latin_correction,
hunspell_orig_correction_in_dataset, hunspell_latin_correction_in_dataset,
and the pysc_ equivalents.
"""

import importlib.metadata
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import enchant
import pandas as pd
from spellchecker import SpellChecker

LOGS_DIR = Path("logs")
IDEOGRAPHIC_SCRIPTS = {"CJK", "Hangul", "Hiragana", "Katakana"}


def _find_parquet() -> Path:
    """Locate the input parquet, tolerating different artifact download layouts."""
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

    # Last resort: search the entire workspace
    found = sorted(Path(".").rglob("*.parquet"))
    log.info("Parquet search found: %s", [str(f) for f in found])
    if len(found) == 1:
        log.info("Using: %s", found[0])
        return found[0]
    raise FileNotFoundError(
        f"No input parquet found. Searched candidates and workspace. "
        f"All .parquet files found: {[str(f) for f in found]}"
    )


def setup_logging(timestamp):
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / ("corrections_" + timestamp + ".log")
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = setup_logging(timestamp)

    log.info("=" * 70)
    log.info("corrections_names.py  —  %s", timestamp)
    log.info("=" * 70)
    log.info("pyenchant %s  |  pyspellchecker %s",
             enchant.__version__, _pkg("pyspellchecker"))

    parquet_path = _find_parquet()
    output_path = Path("data/advanced_results_base.parquet")
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

    d_hunspell = enchant.Dict("en_US")
    spell_pysc  = SpellChecker()

    # hunspell corrections
    log.info("")
    log.info("[hunspell corrections]")
    h_unknowns = (
        set(df.loc[~df["hunspell_orig_known"],  "name"].dropna().unique()) |
        set(df.loc[~df["hunspell_latin_known"], "name_latin"].dropna().unique())
    ) - ideographic_words
    log.info("  %s unique unknowns (combined A+B, ideographic excluded)",
             f"{len(h_unknowns):,}")
    h_corr_map = hunspell_corrections(h_unknowns, d_hunspell)

    df["hunspell_orig_correction"]  = df["name"].map(h_corr_map)
    df["hunspell_latin_correction"] = df["name_latin"].map(h_corr_map)
    df.loc[ideographic_mask, "hunspell_orig_correction"] = None
    df["hunspell_orig_correction_in_dataset"]  = df["hunspell_orig_correction"].str.lower().isin(dataset_latin_set)
    df["hunspell_latin_correction_in_dataset"] = df["hunspell_latin_correction"].str.lower().isin(dataset_latin_set)

    log.info("Checkpoint: saving after hunspell corrections…")
    df.to_parquet(output_path, index=False)
    log.info("Checkpoint saved.")

    # pyspellchecker corrections
    log.info("")
    log.info("[pyspellchecker corrections]")
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

    log.info("")
    log.info("Saving to %s…", output_path)
    df.to_parquet(output_path, index=False)
    log.info("Done.  Log: %s", log_path)


if __name__ == "__main__":
    main()
