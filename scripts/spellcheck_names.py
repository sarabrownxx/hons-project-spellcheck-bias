"""
scripts/spellcheck_names.py

Measures algorithmic bias in English spell-checkers against names from
different origins. Requires preprocess_names.py to have been run first
(name_script and name_latin columns must exist in the parquet).

Research design
───────────────
Each name is checked under two conditions:

  Condition A — original script:  the name as stored, which may be
      Arabic, CJK, Cyrillic, Latin, etc.
  Condition B — Latin/ASCII:      the anyascii transliteration (name_latin),
      so every name is in a form the spell-checkers were designed to handle.

Comparing A vs B isolates script-level bias.  Comparing Western Latin names
against non-Western Latin names under Condition B isolates lexical/phonological
bias independently of script.

Spell-checkers used
───────────────────
  hunspell    (via pyenchant, en_US dictionary) — primary tool.  The engine
              behind LibreOffice, Firefox, macOS, and Chrome.  Real-world
              impact: this is what users actually encounter.

  pyspellchecker  (Norvig edit-distance + English frequency dict) — secondary
              baseline.  Algorithmically simpler; useful for comparison and
              as a sanity check on the hunspell findings.

New columns added
─────────────────
  hunspell_orig_known       hunspell: original name recognised (en_US)
  hunspell_orig_correction  hunspell: top suggestion when not recognised
  hunspell_latin_known      hunspell: name_latin recognised
  hunspell_latin_correction hunspell: top suggestion for name_latin
  hunspell_orig_correction_in_dataset   correction matches a dataset name
  hunspell_latin_correction_in_dataset  correction matches a dataset name

  pysc_orig_known           pyspellchecker: original name recognised
  pysc_orig_correction      pyspellchecker: top correction for original
  pysc_latin_known          pyspellchecker: name_latin recognised
  pysc_latin_correction     pyspellchecker: top correction for name_latin
  pysc_orig_correction_in_dataset
  pysc_latin_correction_in_dataset

Note: *_orig_correction is set to None for CJK/Hangul/Hiragana/Katakana
originals because single-codepoint characters produce spurious short-word
corrections (e.g. a 2-char Chinese name matches English "i" at edit distance 1).

Usage
─────
  python scripts/spellcheck_names.py

Outputs
───────
  data/names_base.parquet        enriched in-place
  logs/spellcheck_<ts>.log       full run log
  logs/spellcheck_<ts>_report.md methodology + results report
"""

import importlib.metadata
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import numpy as np
import pandas as pd
import enchant
from spellchecker import SpellChecker

# ── Configuration ──────────────────────────────────────────────────────────────

PARQUET_PATH = Path("data/names_base.parquet")
LOGS_DIR     = Path("logs")

# Scripts where a single Unicode codepoint represents one glyph/syllable,
# making Condition A corrections meaningless (spurious edit-distance matches).
IDEOGRAPHIC_SCRIPTS = {"CJK", "Hangul", "Hiragana", "Katakana"}

# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging(timestamp: str) -> Path:
    LOGS_DIR.mkdir(exist_ok=True)
    log_path = LOGS_DIR / ("spellcheck_" + timestamp + ".log")
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


# ── Checker implementations ────────────────────────────────────────────────────

def hunspell_batch_known(words: list, d: enchant.Dict) -> set:
    """Return the subset of words that hunspell (en_US) considers correct."""
    known = set()
    for w in words:
        if w:
            try:
                if d.check(w):
                    known.add(w)
            except enchant.errors.Error:
                pass
    return known


def hunspell_corrections(unknowns: set, d: enchant.Dict) -> dict:
    """Return {word: top_suggestion_or_None} for each unknown word."""
    result = {}
    for w in unknowns:
        if w:
            try:
                suggestions = d.suggest(w)
                result[w] = suggestions[0] if suggestions else None
            except enchant.errors.Error:
                result[w] = None
    return result


def pysc_batch_known(words: list, spell: SpellChecker) -> set:
    """Return the subset of words that pyspellchecker considers known."""
    return spell.known(words)


def pysc_corrections(unknowns: set, spell: SpellChecker) -> dict:
    """Return {word: correction_or_None} for each unknown word."""
    return {w: spell.correction(w) for w in unknowns if w}


# ── Core check runner ──────────────────────────────────────────────────────────

def run_checker(df: pd.DataFrame,
                name_col: str,
                batch_known_fn,
                corrections_fn,
                prefix: str,
                dataset_latin_set: set,
                null_ideographic: bool = False) -> pd.DataFrame:
    """
    Run one spell-checker on one name column (one condition).
    Adds {prefix}_known and {prefix}_correction columns to df.
    If null_ideographic is True, nulls corrections for ideographic-script names.
    """
    unique_words = df[name_col].unique().tolist()
    log.info("  %s — checking %s unique values…", prefix, f"{len(unique_words):,}")

    known = batch_known_fn(unique_words)
    unknown = set(unique_words) - known

    log.info("    Known: %s  Unknown: %s  (%.1f%% known)",
             f"{len(known):,}", f"{len(unknown):,}",
             100 * len(known) / max(len(unique_words), 1))
    log.info("    Computing corrections for %s unknowns…", f"{len(unknown):,}")

    corr_map = corrections_fn(unknown)

    df[prefix + "_known"]      = df[name_col].isin(known)
    df[prefix + "_correction"] = df[name_col].map(corr_map)

    if null_ideographic:
        mask = df["name_script"].isin(IDEOGRAPHIC_SCRIPTS)
        df.loc[mask, prefix + "_correction"] = None
        log.info("    Nulled corrections for %s ideographic-script names",
                 f"{mask.sum():,}")

    # Cross-condition: does the correction point to a real dataset name?
    df[prefix + "_correction_in_dataset"] = (
        df[prefix + "_correction"]
          .str.lower()
          .isin(dataset_latin_set)
    )

    return df


# ── Run all checkers ───────────────────────────────────────────────────────────

def run_all_checkers(df: pd.DataFrame,
                     dataset_latin_set: set) -> tuple:
    """
    Runs hunspell and pyspellchecker under both conditions (A and B).
    Returns the enriched DataFrame and a stats dict.
    """
    t0 = time.time()

    d_hunspell = enchant.Dict("en_US")
    spell_pysc  = SpellChecker()

    log.info("")
    log.info("[hunspell — Condition A] original names")
    df = run_checker(df, "name",
                     lambda words: hunspell_batch_known(words, d_hunspell),
                     lambda unknowns: hunspell_corrections(unknowns, d_hunspell),
                     "hunspell_orig", dataset_latin_set,
                     null_ideographic=True)

    log.info("[hunspell — Condition B] name_latin")
    df = run_checker(df, "name_latin",
                     lambda words: hunspell_batch_known(words, d_hunspell),
                     lambda unknowns: hunspell_corrections(unknowns, d_hunspell),
                     "hunspell_latin", dataset_latin_set)

    # Checkpoint: hunspell results safe on disk before the pyspellchecker pass
    log.info("Checkpoint: saving hunspell results…")
    df.to_parquet(PARQUET_PATH, index=False)
    log.info("Checkpoint saved.")

    log.info("[pyspellchecker — Condition A] original names")
    df = run_checker(df, "name",
                     lambda words: pysc_batch_known(words, spell_pysc),
                     lambda unknowns: pysc_corrections(unknowns, spell_pysc),
                     "pysc_orig", dataset_latin_set,
                     null_ideographic=True)

    log.info("[pyspellchecker — Condition B] name_latin")
    df = run_checker(df, "name_latin",
                     lambda words: pysc_batch_known(words, spell_pysc),
                     lambda unknowns: pysc_corrections(unknowns, spell_pysc),
                     "pysc_latin", dataset_latin_set)

    duration = time.time() - t0
    log.info("All checkers complete in %.0fs", duration)

    # ── Summary stats ─────────────────────────────────────────────────────────
    def pct(col): return round(100 * df[col].mean(), 2)

    stats = {
        "duration_s":   round(duration, 1),
        "n_total":      len(df),
        "hunspell": {
            "version":    enchant.__version__,
            "dictionary": "en_US",
            "pct_orig_known":   pct("hunspell_orig_known"),
            "pct_latin_known":  pct("hunspell_latin_known"),
            "n_orig_correction_in_dataset":
                int(df["hunspell_orig_correction_in_dataset"].sum()),
            "n_latin_correction_in_dataset":
                int(df["hunspell_latin_correction_in_dataset"].sum()),
        },
        "pysc": {
            "version":   _pkg("pyspellchecker"),
            "language":  "en",
            "pct_orig_known":   pct("pysc_orig_known"),
            "pct_latin_known":  pct("pysc_latin_known"),
            "n_orig_correction_in_dataset":
                int(df["pysc_orig_correction_in_dataset"].sum()),
            "n_latin_correction_in_dataset":
                int(df["pysc_latin_correction_in_dataset"].sum()),
        },
        "script_breakdown":  _script_breakdown(df),
        "country_breakdown": _country_breakdown(df),
    }

    for tool, s in [("hunspell", stats["hunspell"]), ("pysc", stats["pysc"])]:
        log.info("  %-18s orig known: %.1f%%  latin known: %.1f%%",
                 tool, s["pct_orig_known"], s["pct_latin_known"])

    return df, stats


def _script_breakdown(df: pd.DataFrame) -> dict:
    result = {}
    for script, grp in df.groupby("name_script"):
        n = len(grp)
        result[script] = {
            "n": n,
            "hunspell_pct_orig_known":   round(100 * grp["hunspell_orig_known"].mean(), 2),
            "hunspell_pct_latin_known":  round(100 * grp["hunspell_latin_known"].mean(), 2),
            "pysc_pct_orig_known":       round(100 * grp["pysc_orig_known"].mean(), 2),
            "pysc_pct_latin_known":      round(100 * grp["pysc_latin_known"].mean(), 2),
        }
    return result


def _country_breakdown(df: pd.DataFrame) -> dict:
    result = {}
    for country, grp in df.groupby("top_country"):
        result[country] = {
            "n": len(grp),
            "hunspell_pct_orig_known":  round(100 * grp["hunspell_orig_known"].mean(), 2),
            "hunspell_pct_latin_known": round(100 * grp["hunspell_latin_known"].mean(), 2),
            "pysc_pct_orig_known":      round(100 * grp["pysc_orig_known"].mean(), 2),
            "pysc_pct_latin_known":     round(100 * grp["pysc_latin_known"].mean(), 2),
        }
    return result


# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(run_meta: dict, stats: dict, report_path: Path) -> None:
    lines = []

    def h(n, t):  lines.append("\n" + "#" * n + " " + t + "\n")
    def p(t):     lines.append(t + "\n")
    def table(headers, rows):
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(c) for c in row) + " |")
        lines.append("")

    h(1, "Spell-Check Bias Analysis — Run Report")
    p("**Generated:** " + run_meta["timestamp"])
    p("**Script:** `scripts/spellcheck_names.py`")
    p("---")

    h(2, "1. Environment")
    p("**Python:** " + run_meta["python_version"])
    for k, v in run_meta["packages"].items():
        p("- `" + k + "` " + v)

    h(2, "2. Research Design")
    p("The research question is whether English spell-check tools exhibit "
      "algorithmic bias against names from non-Western or non-Latin-script "
      "origins. Each name is processed under two conditions:")
    p("- **Condition A (original):** the name as stored — may be Arabic, CJK, "
      "Cyrillic, Latin, etc.")
    p("- **Condition B (transliterated):** the `name_latin` column produced by "
      "`anyascii` in `preprocess_names.py`, so all names are in a Latin/ASCII "
      "form the spell-checkers were designed to process.")
    p("Two independent spell-checkers are used so findings can be corroborated "
      "or contrasted across tools with different underlying algorithms.")

    h(2, "3. Spell-Checkers")
    h(3, "3.1 hunspell (primary)")
    p("- **Library:** `pyenchant` v" + stats["hunspell"]["version"] +
      ", dictionary: " + stats["hunspell"]["dictionary"])
    p("- **Rationale:** hunspell is the engine behind LibreOffice, Firefox, "
      "macOS system spell check, and Chrome. It is the spell-checker that real "
      "users encounter daily. Bias findings here have direct real-world "
      "significance.")
    p("- **API used:** `enchant.Dict.check(word)` (known/unknown), "
      "`enchant.Dict.suggest(word)` (top correction).")

    h(3, "3.2 pyspellchecker (secondary baseline)")
    p("- **Library:** `pyspellchecker` v" + stats["pysc"]["version"])
    p("- **Rationale:** Uses a pre-built English word-frequency dictionary "
      "with Damerau-Levenshtein edit-distance candidate generation (Norvig "
      "approach). Algorithmically distinct from hunspell, enabling comparison "
      "of bias across different spell-check strategies.")

    h(3, "3.3 Ideographic script correction nulling")
    p("For names in CJK, Hangul, Hiragana, and Katakana scripts, each "
      "character occupies a single Unicode codepoint. A 2-character Chinese "
      "name therefore has string length 2, making spurious corrections "
      "appear at edit distance 1 (e.g. 'i', 'oo'). Condition A corrections "
      "are set to None for these scripts. Condition B is unaffected as "
      "anyascii produces longer Latin strings.")

    h(2, "4. Results — Overall")
    table(
        ["Tool", "Condition", "% names recognised"],
        [
            ["hunspell",         "A — original script",
             str(stats["hunspell"]["pct_orig_known"]) + "%"],
            ["hunspell",         "B — anyascii (Latin)",
             str(stats["hunspell"]["pct_latin_known"]) + "%"],
            ["pyspellchecker",   "A — original script",
             str(stats["pysc"]["pct_orig_known"]) + "%"],
            ["pyspellchecker",   "B — anyascii (Latin)",
             str(stats["pysc"]["pct_latin_known"]) + "%"],
        ]
    )
    p("A gap between Condition A and B for non-Latin scripts indicates "
      "script-level bias. A persistent gap between Western and non-Western "
      "Latin names under Condition B indicates lexical/phonological bias.")

    n = stats["n_total"]
    p("**Corrections pointing to a dataset name:**")
    table(
        ["Tool", "Condition", "n corrections in dataset"],
        [
            ["hunspell",       "A", str(stats["hunspell"]["n_orig_correction_in_dataset"])],
            ["hunspell",       "B", str(stats["hunspell"]["n_latin_correction_in_dataset"])],
            ["pyspellchecker", "A", str(stats["pysc"]["n_orig_correction_in_dataset"])],
            ["pyspellchecker", "B", str(stats["pysc"]["n_latin_correction_in_dataset"])],
        ]
    )
    p("When a correction matches a name in the dataset it suggests the "
      "spell-checker is steering the input toward a recognised name, rather "
      "than returning an arbitrary English word.")

    h(2, "5. Results — Breakdown by Script")
    table(
        ["Script", "n",
         "hn orig%", "hn latin%", "hn Δ",
         "py orig%", "py latin%", "py Δ"],
        [
            (s,
             f"{d['n']:,}",
             f"{d['hunspell_pct_orig_known']:.1f}%",
             f"{d['hunspell_pct_latin_known']:.1f}%",
             f"{d['hunspell_pct_latin_known'] - d['hunspell_pct_orig_known']:+.1f}pp",
             f"{d['pysc_pct_orig_known']:.1f}%",
             f"{d['pysc_pct_latin_known']:.1f}%",
             f"{d['pysc_pct_latin_known'] - d['pysc_pct_orig_known']:+.1f}pp")
            for s, d in sorted(stats["script_breakdown"].items(),
                                key=lambda x: -x[1]["n"])
        ]
    )
    p("hn = hunspell, py = pyspellchecker, Δ = Condition B minus Condition A.")

    h(2, "6. Results — Breakdown by Country of Origin")
    p("Recognition rates per `top_country`, sorted by name count descending.")
    table(
        ["Country", "n", "hn orig%", "hn latin%", "py orig%", "py latin%"],
        [
            (c, f"{d['n']:,}",
             f"{d['hunspell_pct_orig_known']:.1f}%",
             f"{d['hunspell_pct_latin_known']:.1f}%",
             f"{d['pysc_pct_orig_known']:.1f}%",
             f"{d['pysc_pct_latin_known']:.1f}%")
            for c, d in sorted(stats["country_breakdown"].items(),
                                key=lambda x: -x[1]["n"])
        ]
    )

    h(2, "7. Column Definitions")
    table(
        ["Column", "Type", "Description"],
        [
            ["hunspell_orig_known",    "bool",
             "hunspell en_US: original name is in dictionary"],
            ["hunspell_orig_correction", "str|None",
             "hunspell: top suggestion; None if known, no match, or ideographic"],
            ["hunspell_latin_known",   "bool",
             "hunspell en_US: name_latin is in dictionary"],
            ["hunspell_latin_correction", "str|None",
             "hunspell: top suggestion for name_latin"],
            ["hunspell_orig_correction_in_dataset", "bool",
             "Condition A correction matches a name in this dataset"],
            ["hunspell_latin_correction_in_dataset", "bool",
             "Condition B correction matches a name in this dataset"],
            ["pysc_orig_known",    "bool",
             "pyspellchecker: original name is recognised"],
            ["pysc_orig_correction", "str|None",
             "pyspellchecker: top correction; None if known or no match"],
            ["pysc_latin_known",   "bool",
             "pyspellchecker: name_latin is recognised"],
            ["pysc_latin_correction", "str|None",
             "pyspellchecker: top correction for name_latin"],
            ["pysc_orig_correction_in_dataset",  "bool",
             "Condition A correction matches a name in this dataset"],
            ["pysc_latin_correction_in_dataset", "bool",
             "Condition B correction matches a name in this dataset"],
        ]
    )

    h(2, "8. Limitations")
    p("1. Both spell-checkers use English-language dictionaries. Recognition "
      "of a name as 'known' primarily reflects whether it also happens to be "
      "a common English word (e.g. 'Grace', 'James') rather than global name "
      "recognition.")
    p("2. anyascii transliteration is approximate for abjad scripts (Arabic, "
      "Hebrew). Arabic 'محمد' → 'mhmd' (consonants only), not the standard "
      "romanisation 'Muhammad'. Condition B may therefore understate "
      "recognition for these scripts.")
    p("3. Condition A corrections are nulled for CJK, Hangul, Hiragana, and "
      "Katakana. See Section 3.3.")
    p("4. Names containing spaces are passed as single tokens and will not be "
      "recognised regardless of origin.")

    h(2, "9. Output Files")
    p("- **Enriched dataset:** `" + str(PARQUET_PATH) + "`")
    p("- **This report:** `" + str(report_path) + "`")
    p("- **Run log:** `" + str(run_meta["log_path"]) + "`")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s", report_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR.mkdir(exist_ok=True)
    log_path    = setup_logging(timestamp)
    report_path = LOGS_DIR / ("spellcheck_" + timestamp + "_report.md")

    log.info("=" * 70)
    log.info("spellcheck_names.py  —  %s", timestamp)
    log.info("=" * 70)

    packages = {
        "pyenchant":        enchant.__version__,
        "pyspellchecker":   _pkg("pyspellchecker"),
        "anyascii":         _pkg("anyascii"),
        "pandas":           _pkg("pandas"),
    }
    run_meta = {
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "packages":       packages,
        "log_path":       str(log_path),
    }

    # ── Load ──────────────────────────────────────────────────────────────────
    log.info("Loading %s…", PARQUET_PATH)
    df = pd.read_parquet(PARQUET_PATH)
    log.info("  %s names loaded", f"{len(df):,}")

    missing = [c for c in ("name_script", "name_latin") if c not in df.columns]
    if missing:
        log.error("Missing columns: %s — run preprocess_names.py first.", missing)
        sys.exit(1)

    # Build dataset lookup for the "correction in dataset" check
    dataset_latin_set = set(df["name_latin"].str.lower().dropna())
    log.info("  Dataset lookup set: %s unique values",
             f"{len(dataset_latin_set):,}")

    # ── Run all checkers ──────────────────────────────────────────────────────
    df, stats = run_all_checkers(df, dataset_latin_set)

    # ── Save ──────────────────────────────────────────────────────────────────
    log.info("")
    log.info("Saving to %s…", PARQUET_PATH)
    df.to_parquet(PARQUET_PATH, index=False)
    log.info("Saved.")

    # ── Report ────────────────────────────────────────────────────────────────
    write_report(run_meta, stats, report_path)

    log.info("")
    log.info("Done.  Log: %s", log_path)
    log.info("       Report: %s", report_path)


if __name__ == "__main__":
    main()
