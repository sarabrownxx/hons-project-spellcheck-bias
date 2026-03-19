"""
Quick local trial of SymSpell, LanguageTool, and Aspell as alternatives to hunspell.

SymSpell      - symmetric delete + frequency dict, pure Python, no system deps
LanguageTool  - Morfologik dict + grammar rule engine, context-aware sentences
                Requires Java 17+. First run downloads ~200 MB.
                While upgrading Java, pass remote_server='https://api.languagetool.org'
                to use the public API instead (rate-limited, fine for a trial).
Aspell        - dictionary-based like hunspell; needs system libaspell (see below)

Same 5 test names as the pysc sandbox:
  James, Grace, Fatima, Priya, Oluwaseun
"""

import importlib.resources
import language_tool_python
from symspellpy import SymSpell, Verbosity

NAMES = [
    ("James",     "Western Latin, common English word, likely known"),
    ("Grace",     "Western Latin, English word + name, likely known"),
    ("Fatima",    "Arabic origin, likely unknown"),
    ("Priya",     "Indian origin, likely unknown"),
    ("Oluwaseun", "Nigerian/Yoruba origin, likely unknown"),
]

def sentences(name):
    return {
        "standalone":     name,
        "sentence_end":   f"My name is {name}",
        "sentence_start": f"{name} is my name",
        "sentence_mid":   f"My name is {name} and that is a fact",
    }


# ---- SymSpell ----------------------------------------------------------------

def load_symspell():
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    pkg = importlib.resources.files("symspellpy")
    with importlib.resources.as_file(pkg / "frequency_dictionary_en_82_765.txt") as p:
        sym.load_dictionary(str(p), term_index=0, count_index=1)
    with importlib.resources.as_file(pkg / "frequency_bigramdictionary_en_243_342.txt") as p:
        sym.load_bigram_dictionary(str(p), term_index=0, count_index=2)
    return sym


def symspell_word(word, sym):
    # include_unknown=True so we always get something back even for no-match
    hits = sym.lookup(word, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)
    if not hits:
        return {"known": False, "correction": None, "distance": None, "frequency": None, "candidates": []}
    top = hits[0]
    return {
        "known":      top.distance == 0,
        "correction": top.term,
        "distance":   top.distance,
        "frequency":  top.count,
        # SuggestItem fields: .term, .distance, .count (corpus frequency)
        "candidates": [(h.term, h.distance, h.count) for h in hits[:5]],
    }


def symspell_compound(sentence, sym):
    # lookup_compound corrects each word using bigram context — potentially
    # context-sensitive unlike single-word lookup
    results = sym.lookup_compound(sentence, max_edit_distance=2)
    return results[0].term if results else sentence


def run_symspell(sym):
    print("\n=== SYMSPELL ===")
    print("algorithm : symmetric delete + frequency ranking")
    print("dictionary: bundled English unigram (82,765 words) + bigram\n")

    for name, desc in NAMES:
        print(f"  {name!r}  ({desc})")

        r = symspell_word(name.lower(), sym)
        if r["known"]:
            print(f"    word lookup : known  (freq={r['frequency']:,})")
        else:
            print(f"    word lookup : UNKNOWN  correction={r['correction']!r}  dist={r['distance']}  freq={r['frequency']:,}")
            print(f"    candidates  : {r['candidates']}")

        print(f"    compound lookup (bigram context):")
        for fmt, sent in sentences(name).items():
            corrected = symspell_compound(sent, sym)
            print(f"      [{fmt:<16}] {sent!r}  ->  {corrected!r}")
        print()


# ---- LanguageTool ------------------------------------------------------------

def lt_spelling_matches(text, tool):
    # filter to misspelling-type matches only, ignoring grammar/style
    return [
        {
            "word":         text[m.offset: m.offset + m.error_length],
            "rule_id":      m.rule_id,
            "message":      m.message,
            "replacements": m.replacements[:3],
            "offset":       m.offset,
        }
        for m in tool.check(text)
        if m.rule_issue_type == "misspelling"
    ]


def run_languagetool(tool):
    print("\n=== LANGUAGETOOL ===")
    print(f"version   : {tool.language_tool_download_version}")
    print("algorithm : Morfologik dictionary + grammar rule engine")
    print("context   : full sentence — rules can interact with surrounding words\n")

    for name, desc in NAMES:
        print(f"  {name!r}  ({desc})")
        for fmt, sent in sentences(name).items():
            flags = lt_spelling_matches(sent, tool)
            if not flags:
                print(f"    [{fmt:<16}] {sent!r}  ->  no spelling flags")
            else:
                for f in flags:
                    print(f"    [{fmt:<16}] {sent!r}")
                    print(f"      flagged: {f['word']!r}  rule={f['rule_id']}")
                    print(f"      message: {f['message']}")
                    print(f"      suggestions: {f['replacements']}")
        print()

    # show raw Match fields for reference
    print("  Raw Match fields (example: 'fatima' standalone)")
    raw = tool.check("fatima")
    if raw:
        m = raw[0]
        for attr in ("rule_id", "rule_issue_type", "message", "replacements",
                     "offset", "error_length", "context", "offset_in_context", "sentence"):
            print(f"    {attr:<18}: {getattr(m, attr)!r}")
    else:
        print("    (no matches for 'fatima')")


# ---- Aspell note -------------------------------------------------------------
#
# Aspell uses a similar dictionary-lookup model to hunspell.  The comparison
# value is that it uses a different scoring algorithm for suggestions (based
# on phonetic similarity via Soundex/Metaphone rather than pure edit distance).
#
# On Ubuntu (GitHub Actions):
#   apt-get install aspell aspell-en
#   pip install aspell-python
#   import aspell
#   s = aspell.Speller('lang', 'en')
#   s.check("Fatima")      # -> bool
#   s.suggest("Fatima")    # -> list[str]
#
# On Windows:
#   1. Download GNU Aspell Win32 installer from http://aspell.net/win32/
#   2. Install the en dictionary package separately
#   3. pip install aspell-python
#   The pyenchant library also supports aspell as a backend
#   (enchant.Dict("en_US", "aspell")) if both the system lib and the
#   aspell dict are installed.
#
# Because of the system dependency this is better trialled on GitHub Actions
# than locally on Windows.  The API shape and output format are essentially
# the same as hunspell — results would slot into the pipeline identically.


# ---- main --------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sandbox: SymSpell and LanguageTool trials")
    parser.add_argument("--tool", choices=["symspell", "languagetool", "both"], default="both",
                        help="Which tool to run (default: both)")
    args = parser.parse_args()

    if args.tool in ("symspell", "both"):
        print("Loading SymSpell...")
        sym = load_symspell()
        print("Ready.")
        run_symspell(sym)

    if args.tool in ("languagetool", "both"):
        print("Loading LanguageTool en-US (downloads ~200 MB on first run)...")
        tool = language_tool_python.LanguageTool("en-US")
        print("Ready.")
        run_languagetool(tool)
        tool.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
