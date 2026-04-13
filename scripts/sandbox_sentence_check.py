"""
sandbox_sentence_check.py

Quick local test to explore whether hunspell's known/unknown result or
suggestion differs depending on whether a name appears:

  Format 0 — standalone    : "Fatima"
  Format 1 — sentence-end  : "My name is Fatima"
  Format 2 — sentence-start: "Fatima is my name"
  Format 3 — sentence-mid  : "My name is Fatima and that is a fact"

Also tests a cleaned name filter: skip names that contain digits, or any
non-letter character other than hyphens, apostrophes, and accented/non-Latin
Unicode letters (i.e. strip names with *, ., numbers, underscores, etc.).

Run from project root:
    python scripts/sandbox_sentence_check.py
"""

import re
import unicodedata
import enchant

# Sample names
# Hand-picked to cover different scripts and origins.
# (name, description)
SAMPLE = [
    # Latin — Western / English-recognisable
    ("James",       "Western Latin — common English name"),
    ("Grace",       "Western Latin — common English word+name"),
    ("Charlotte",   "Western Latin — common English name"),
    # Latin — non-Western origin, Latin letters
    ("Fatima",      "Latin — Arabic origin"),
    ("Mohammed",    "Latin — Arabic origin"),
    ("Yuki",        "Latin — Japanese origin"),
    ("Priya",       "Latin — Indian origin"),
    ("Oluwaseun",   "Latin — Nigerian/Yoruba origin"),
    ("Nguyễn",      "Latin — Vietnamese (accented)"),
    # Latin — hyphenated / apostrophe
    ("Mary-Jane",   "Latin — hyphenated"),
    ("O'Brien",     "Latin — apostrophe"),
    # Non-Latin scripts
    ("محمد",        "Arabic script"),
    ("张伟",         "CJK script"),
    ("Андрей",      "Cyrillic script"),
    ("देवेंद्र",   "Devanagari script"),
    # Noisy / should be filtered
    ("J4mes",       "Contains digit — should be filtered"),
    ("A*B",         "Contains * — should be filtered"),
    ("A.B",         "Contains . — should be filtered (not a name separator)"),
    ("A A",         "All single-char parts — already filtered at build time"),
]

# Sentence templates
def make_sentences(name: str) -> dict:
    return {
        "standalone":   name,
        "sentence_end": f"My name is {name}",
        "sentence_start": f"{name} is my name",
        "sentence_mid": f"My name is {name} and that is a fact",
    }

# Name cleaner filter
_ALLOWED_NON_LETTER = set("-' ")   # hyphen, apostrophe, space (for multiword)

def is_clean(name: str) -> bool:
    """
    Return True if name passes the noise filter:
      - no digits
      - no characters other than Unicode letters, hyphen, apostrophe, space
      - not all single-char space-separated parts (already handled upstream)
    """
    for ch in name:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):   # any Unicode letter — allow
            continue
        if cat.startswith("M"):   # combining marks (e.g. Devanagari anusvara) — allow
            continue
        if ch in _ALLOWED_NON_LETTER:
            continue
        return False  # digit, punctuation, symbol, etc.
    return True

# Hunspell checkers
def check_word(word: str, d: enchant.Dict) -> dict:
    """Check a single word string directly — the baseline."""
    try:
        known = d.check(word)
        suggs = d.suggest(word) if not known else []
    except enchant.errors.Error:
        known, suggs = False, []
    return {"known": known, "suggestions": suggs[:3]}


def check_whole_string(sentence: str, d: enchant.Dict) -> dict:
    """
    Pass the ENTIRE sentence string to d.check() / d.suggest() as-is —
    no splitting.  Hunspell treats the whole string as one 'word', so
    check() will almost certainly return False and suggest() will produce
    whatever edit-distance candidates it finds for the full string.
    Shows what hunspell actually does when given a raw sentence.
    """
    try:
        known = d.check(sentence)
        suggs = d.suggest(sentence) if not known else []
    except enchant.errors.Error:
        known, suggs = False, []
    return {"known": known, "suggestions": suggs[:3]}


def check_tokens(sentence: str, d: enchant.Dict) -> list:
    """
    Split sentence by whitespace, check each token individually.
    This is functionally identical to checking the name standalone —
    the result for the name token will match check_word() exactly.
    Included to make this equivalence explicit.
    """
    results = []
    for tok in sentence.split():
        stripped = tok.strip(".,!?;:")
        if not stripped:
            continue
        try:
            known = d.check(stripped)
            suggs = d.suggest(stripped) if not known else []
        except enchant.errors.Error:
            known, suggs = False, []
        results.append({"token": stripped, "known": known, "suggestions": suggs[:3]})
    return results


# Pretty printer
def print_name_results(name: str, desc: str, d: enchant.Dict, clean: bool):
    print(f"\n{'='*70}")
    print(f"  Name: {name!r}  ({desc})")
    print(f"  Clean filter: {'PASS' if clean else 'FILTERED OUT'}")
    print(f"{'='*70}")

    if not clean:
        print("  [skipped — does not pass noise filter]")
        return

    sentences = make_sentences(name)

    # Approach A: pass the full sentence string directly to hunspell
    print("\n  APPROACH A — full sentence string passed directly to d.check() / d.suggest()")
    print("  (hunspell treats the whole string as one word)")
    for fmt, sentence in sentences.items():
        r = check_whole_string(sentence, d)
        status = "OK " if r["known"] else "???"
        sugg_str = f"  → {r['suggestions']}" if not r["known"] else ""
        print(f"    {status}  {sentence!r}{sugg_str}")

    # Approach B: split into tokens, check each individually
    print("\n  APPROACH B — sentence split into tokens, each checked separately")
    print("  (shows that the name token result == standalone result)")
    for fmt, sentence in sentences.items():
        tokens = check_tokens(sentence, d)
        flagged = [(t["token"], t["suggestions"]) for t in tokens if not t["known"]]
        flagged_str = ", ".join(
            f"{tok!r} → {sugg}" if sugg else f"{tok!r} (no suggestions)"
            for tok, sugg in flagged
        ) if flagged else "all tokens known"
        print(f"    [{fmt}]  flagged: {flagged_str}")

# Main
def main():
    print("Initialising hunspell en_US…")
    d = enchant.Dict("en_US")
    print("Ready.\n")

    for name, desc in SAMPLE:
        clean = is_clean(name)
        print_name_results(name, desc, d, clean)

    print(f"\n{'='*70}")
    print("Done.")

if __name__ == "__main__":
    main()
