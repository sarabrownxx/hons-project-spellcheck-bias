"""
sandbox_pysc_sentence_check.py

Explores whether pyspellchecker's correction/known result differs depending
on how a name is presented:

  Format 0 — standalone    : "fatima"  (lowercased, as pysc sees it)
  Format 1 — sentence-end  : "My name is Fatima"  → token "Fatima"
  Format 2 — sentence-start: "Fatima is my name"  → token "Fatima"
  Format 3 — sentence-mid  : "My name is Fatima and that is a fact" → token "Fatima"

pyspellchecker internally lowercases all input, so context position in a
sentence cannot affect the result — but this script makes that explicit and
also checks whether capitalisation of the name token itself changes anything.

Also checks:
  - spell.known()      — does pysc consider the name in its vocabulary?
  - spell.correction() — top correction candidate
  - spell.candidates() — full candidate set (first 5)
"""

from spellchecker import SpellChecker

# 5 representative names
# Chosen to span Western-known, non-Western Latin, hyphenated, and
# a common English word that is also a name (Grace).
NAMES = [
    ("James",     "Western Latin — common English name (likely known)"),
    ("Grace",     "Western Latin — English word + name (likely known)"),
    ("Fatima",    "Latin — Arabic origin (likely unknown)"),
    ("Priya",     "Latin — Indian origin (likely unknown)"),
    ("Oluwaseun", "Latin — Nigerian/Yoruba origin (likely unknown)"),
]

# Sentence templates
def make_sentences(name: str) -> dict:
    return {
        "standalone":     name,
        "sentence_end":   f"My name is {name}",
        "sentence_start": f"{name} is my name",
        "sentence_mid":   f"My name is {name} and that is a fact",
    }


# Helpers
def extract_name_token(sentence: str, name: str) -> str:
    """Return the token from the sentence that matches the name (case-insensitive)."""
    for tok in sentence.split():
        tok_stripped = tok.strip(".,!?;:")
        if tok_stripped.lower() == name.lower():
            return tok_stripped
    return name  # fallback: name is the whole sentence (standalone)


def check_token(token: str, spell: SpellChecker) -> dict:
    """
    Run pyspellchecker on a single token.
    spell.known() and spell.correction() both lowercase internally,
    so "Fatima" and "fatima" produce identical results.
    """
    known      = bool(spell.known([token]))
    correction = spell.correction(token)
    candidates = sorted(spell.candidates(token) or set())[:5]
    return {
        "token":      token,
        "known":      known,
        "correction": correction,
        "candidates": candidates,
    }


def check_whole_sentence(sentence: str, spell: SpellChecker) -> dict:
    """
    Pass the ENTIRE sentence as a single 'word' to pyspellchecker —
    analogous to the whole-string test in sandbox_sentence_check.py.
    Expected: always unknown; correction will be some short English word.
    """
    known      = bool(spell.known([sentence]))
    correction = spell.correction(sentence)
    candidates = sorted(spell.candidates(sentence) or set())[:3]
    return {
        "known":      known,
        "correction": correction,
        "candidates": candidates,
    }


# Pretty printer
def print_name_results(name: str, desc: str, spell: SpellChecker):
    print(f"\n{'='*72}")
    print(f"  Name: {name!r}  ({desc})")
    print(f"{'='*72}")

    sentences = make_sentences(name)

    # Approach A: whole sentence passed as one 'word'
    print("\n  APPROACH A — full sentence string passed as a single token to pysc")
    print("  (pysc treats multi-word string as one unknown word)")
    for fmt, sentence in sentences.items():
        r = check_whole_sentence(sentence, spell)
        status = "OK " if r["known"] else "???"
        print(f"    {status}  {sentence!r}")
        if not r["known"]:
            print(f"         correction={r['correction']!r}  candidates={r['candidates']}")

    # Approach B: extract name token from each sentence, check it
    print("\n  APPROACH B — name token extracted from sentence, checked individually")
    print("  (expected: identical result regardless of sentence position)")
    results_by_fmt = {}
    for fmt, sentence in sentences.items():
        token = extract_name_token(sentence, name)
        r     = check_token(token, spell)
        results_by_fmt[fmt] = r

    # Print and explicitly flag if any result differs from standalone
    standalone_r = results_by_fmt["standalone"]
    for fmt, r in results_by_fmt.items():
        same = (r["known"] == standalone_r["known"] and
                r["correction"] == standalone_r["correction"] and
                r["candidates"] == standalone_r["candidates"])
        diff_flag = "" if same else "  *** DIFFERS FROM STANDALONE ***"
        status = "known  " if r["known"] else f"UNKNOWN  → correction={r['correction']!r}  candidates={r['candidates']}"
        print(f"    [{fmt:<16}]  token={r['token']!r}  {status}{diff_flag}")

    # Capitalisation sensitivity check
    print("\n  CAPITALISATION CHECK — lower / title / upper variants of the name token")
    for variant in (name.lower(), name.title(), name.upper()):
        r = check_token(variant, spell)
        status = "known  " if r["known"] else f"UNKNOWN  → correction={r['correction']!r}"
        print(f"    {variant!r:<20}  {status}")


# Main
def main():
    print("Initialising pyspellchecker (en)…")
    spell = SpellChecker()
    print("Ready.\n")

    for name, desc in NAMES:
        print_name_results(name, desc, spell)

    print(f"\n{'='*72}")
    print("Summary")
    print(f"{'='*72}")
    print("""
Pyspellchecker lowercases all tokens before lookup/correction, so:
  - Sentence position (start / mid / end) has NO effect on the result.
  - Capitalisation of the name token has NO effect on the result.
  - Whole-sentence strings are always unknown (spaces are not in the vocab).

The only thing that matters is the lowercased token string itself.
This mirrors the finding for hunspell in sandbox_sentence_check.py.
Sentence-context wrapping cannot improve pysc correction rates for names.
""")


if __name__ == "__main__":
    main()
