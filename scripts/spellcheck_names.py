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

New columns added
─────────────────
  hunspell_orig_known   hunspell: original name recognised (en_US)
  hunspell_latin_known  hunspell: name_latin recognised

Correction suggestions are computed separately by corrections_names.py,
which must be run after this script.

Usage
─────
  python scripts/spellcheck_names.py

Outputs
───────
  data/names_results_base.parquet        enriched in-place
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

import pandas as pd
import enchant

# Configuration
PARQUET_PATH  = Path("data/names_results_base.parquet")
LOGS_DIR      = Path("logs")

# Scripts where a single Unicode codepoint represents one glyph/syllable,
# making Condition A corrections meaningless (spurious edit-distance matches).
IDEOGRAPHIC_SCRIPTS = {"CJK", "Hangul", "Hiragana", "Katakana"}

# Logging
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


# Checker implementations
def hunspell_batch_known(words: list, d: enchant.Dict) -> set:
    """Return the subset of words that hunspell (en_US) considers correct."""
    known = set()
    total = len(words)
    t0 = time.time()
    for i, w in enumerate(words):
        if w:
            try:
                if d.check(w):
                    known.add(w)
            except enchant.errors.Error:
                pass
        if (i + 1) % 100_000 == 0:
            log.info("    hunspell check: %s / %s (%.0f%%)  %.0fs elapsed",
                     f"{i+1:,}", f"{total:,}", 100 * (i + 1) / total,
                     time.time() - t0)
    log.info("    hunspell check done — %s known / %s  (%.0fs)",
             f"{len(known):,}", f"{total:,}", time.time() - t0)
    return known


def hunspell_corrections(unknowns: set, d: enchant.Dict) -> dict:
    """Return {word: top_suggestion_or_None} for each unknown word."""
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


# Core check runner
def run_checker(df: pd.DataFrame,
                name_col: str,
                batch_known_fn,
                prefix: str) -> pd.DataFrame:
    """
    Run one spell-checker on one name column (one condition).
    Adds {prefix}_known to df.  Corrections are handled separately by
    corrections_names.py.
    """
    unique_words = df[name_col].unique().tolist()
    log.info("  %s — checking %s unique values…", prefix, f"{len(unique_words):,}")

    known = batch_known_fn(unique_words)
    unknown = set(unique_words) - known

    log.info("    Known: %s  Unknown: %s  (%.1f%% known)",
             f"{len(known):,}", f"{len(unknown):,}",
             100 * len(known) / max(len(unique_words), 1))

    df[prefix + "_known"] = df[name_col].isin(known)
    return df


# Run all checkers
def run_all_checkers(df: pd.DataFrame) -> tuple:
    t0 = time.time()

    d_hunspell = enchant.Dict("en_US")

    cond_a_words = set(df["name"].dropna().unique())
    cond_b_words = set(df["name_latin"].dropna().unique())
    all_words    = list(cond_a_words | cond_b_words)
    overlap      = cond_a_words & cond_b_words

    log.info("Unique words — Condition A: %s  Condition B: %s  "
             "union: %s  overlap (name==name_latin): %s",
             f"{len(cond_a_words):,}", f"{len(cond_b_words):,}",
             f"{len(all_words):,}", f"{len(overlap):,}")

    log.info("")
    log.info("Pre-computing hunspell over %s combined unique words…",
             f"{len(all_words):,}")
    h_known = hunspell_batch_known(all_words, d_hunspell)
    log.info("  hunspell: %s known  %s unknown",
             f"{len(h_known):,}", f"{len(all_words) - len(h_known):,}")

    log.info("[hunspell — Condition A] original names")
    df = run_checker(df, "name", lambda _: h_known, "hunspell_orig")

    log.info("[hunspell — Condition B] name_latin")
    df = run_checker(df, "name_latin", lambda _: h_known, "hunspell_latin")

    duration = time.time() - t0
    log.info("All checkers complete in %.0fs", duration)

    def pct(col): return round(100 * df[col].mean(), 2)

    stats = {
        "duration_s": round(duration, 1),
        "n_total":    len(df),
        "hunspell": {
            "version":         enchant.__version__,
            "dictionary":      "en_US",
            "pct_orig_known":  pct("hunspell_orig_known"),
            "pct_latin_known": pct("hunspell_latin_known"),
        },
        "script_breakdown":  _script_breakdown(df),
        "country_breakdown": _country_breakdown(df),
    }

    log.info("  hunspell  orig known: %.1f%%  latin known: %.1f%%",
             stats["hunspell"]["pct_orig_known"], stats["hunspell"]["pct_latin_known"])

    return df, stats


def _script_breakdown(df: pd.DataFrame) -> dict:
    result = {}
    for script, grp in df.groupby("name_script"):
        n = len(grp)
        result[script] = {
            "n": n,
            "hunspell_pct_orig_known":  round(100 * grp["hunspell_orig_known"].mean(), 2),
            "hunspell_pct_latin_known": round(100 * grp["hunspell_latin_known"].mean(), 2),
        }
    return result


def _country_breakdown(df: pd.DataFrame) -> dict:
    result = {}
    for country, grp in df.groupby("top_country"):
        result[country] = {
            "n": len(grp),
            "hunspell_pct_orig_known":  round(100 * grp["hunspell_orig_known"].mean(), 2),
            "hunspell_pct_latin_known": round(100 * grp["hunspell_latin_known"].mean(), 2),
        }
    return result


# Report
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
    p("hunspell is used to classify each name as known or unknown under each "
      "condition and to generate correction suggestions for unknown names.")

    h(2, "3. Spell-Checker")
    h(3, "3.1 hunspell")
    p("- **Library:** `pyenchant` v" + stats["hunspell"]["version"] +
      ", dictionary: " + stats["hunspell"]["dictionary"])
    p("- **Rationale:** hunspell is the engine behind LibreOffice, Firefox, "
      "macOS system spell check, and Chrome. It is the spell-checker that real "
      "users encounter daily. Bias findings here have direct real-world "
      "significance.")
    p("- **API used:** `enchant.Dict.check(word)` (known/unknown).")

    p("Note: correction suggestions are computed separately by "
      "`corrections_names.py` and are not included in this report.")

    h(2, "4. Results — Overall")
    table(
        ["Condition", "% names recognised"],
        [
            ["A — original script",  str(stats["hunspell"]["pct_orig_known"]) + "%"],
            ["B — anyascii (Latin)", str(stats["hunspell"]["pct_latin_known"]) + "%"],
        ]
    )
    p("A gap between Condition A and B for non-Latin scripts indicates "
      "script-level bias. A persistent gap between Western and non-Western "
      "Latin names under Condition B indicates lexical/phonological bias.")

    h(2, "5. Results — Breakdown by Script")
    table(
        ["Script", "n", "orig%", "latin%", "Δ"],
        [
            (s,
             f"{d['n']:,}",
             f"{d['hunspell_pct_orig_known']:.1f}%",
             f"{d['hunspell_pct_latin_known']:.1f}%",
             f"{d['hunspell_pct_latin_known'] - d['hunspell_pct_orig_known']:+.1f}pp")
            for s, d in sorted(stats["script_breakdown"].items(),
                                key=lambda x: -x[1]["n"])
        ]
    )
    p("Δ = Condition B minus Condition A (hunspell en_US).")

    h(2, "6. Results — Breakdown by Country of Origin")
    p("Recognition rates per `top_country`, sorted by name count descending.")
    table(
        ["Country", "n", "orig%", "latin%"],
        [
            (c, f"{d['n']:,}",
             f"{d['hunspell_pct_orig_known']:.1f}%",
             f"{d['hunspell_pct_latin_known']:.1f}%")
            for c, d in sorted(stats["country_breakdown"].items(),
                                key=lambda x: -x[1]["n"])
        ]
    )

    h(2, "7. Column Definitions")
    table(
        ["Column", "Type", "Description"],
        [
            ["hunspell_orig_known",  "bool",
             "hunspell en_US: original name is in dictionary"],
            ["hunspell_latin_known", "bool",
             "hunspell en_US: name_latin is in dictionary"],
        ]
    )
    p("Correction columns (`*_correction`, `*_correction_match`) are "
      "added by `corrections_names.py`.")

    h(2, "8. Limitations")
    p("1. Both spell-checkers use English-language dictionaries. Recognition "
      "of a name as 'known' primarily reflects whether it also happens to be "
      "a common English word (e.g. 'Grace', 'James') rather than global name "
      "recognition.")
    p("2. anyascii transliteration is approximate for abjad scripts (Arabic, "
      "Hebrew). Arabic 'محمد' → 'mhmd' (consonants only), not the standard "
      "romanisation 'Muhammad'. Condition B may therefore understate "
      "recognition for these scripts.")
    p("3. Names containing spaces are passed as single tokens and will not be "
      "recognised regardless of origin.")

    h(2, "9. Output Files")
    p("- **Enriched dataset:** `" + str(PARQUET_PATH) + "`")
    p("- **This report:** `" + str(report_path) + "`")
    p("- **Run log:** `" + str(run_meta["log_path"]) + "`")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s", report_path)


# Main
def main():
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOGS_DIR.mkdir(exist_ok=True)
    log_path    = setup_logging(timestamp)
    report_path = LOGS_DIR / ("spellcheck_" + timestamp + "_report.md")

    log.info("=" * 70)
    log.info("spellcheck_names.py  —  %s", timestamp)
    log.info("=" * 70)

    packages = {
        "pyenchant": enchant.__version__,
        "anyascii":  _pkg("anyascii"),
        "pandas":    _pkg("pandas"),
    }
    run_meta = {
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "packages":       packages,
        "log_path":       str(log_path),
    }

    # Load
    log.info("Loading %s…", PARQUET_PATH)
    df = pd.read_parquet(PARQUET_PATH)
    log.info("  %s names loaded", f"{len(df):,}")

    missing = [c for c in ("name_script", "name_latin") if c not in df.columns]
    if missing:
        log.error("Missing columns: %s — run preprocess_names.py first.", missing)
        sys.exit(1)

    # Run all checkers
    df, stats = run_all_checkers(df)

    # Save
    log.info("")
    log.info("Saving to %s…", PARQUET_PATH)
    df.to_parquet(PARQUET_PATH, index=False)
    log.info("Saved.")

    # Report
    write_report(run_meta, stats, report_path)

    log.info("")
    log.info("Done.  Log: %s", log_path)
    log.info("       Report: %s", report_path)


if __name__ == "__main__":
    main()
