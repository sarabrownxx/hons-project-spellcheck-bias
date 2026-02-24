# Project Notes — Name Origin & Spell-Check Bias Dataset

A running record of all decisions, methodology, tools, and findings for
reference when writing up the dissertation.

---

## 1. Project Overview

The project has two interconnected goals:

**Goal 1 — Build a validated name-origin dataset.**
Construct a dataset of 727,545 first names, each with a country-of-origin
prediction and an *agreement score* measuring how well that prediction is
corroborated by independent external tools (ethnicolr, langdetect,
nationalize.io).

**Goal 2 — Measure algorithmic bias in English spell-checkers.**
Investigate whether English spell-checking tools disproportionately flag names
from non-Western or non-Latin-script origins as misspellings. This is done
using a *double-run design*: each name is checked twice — once in its original
Unicode script (Condition A) and once as its Latin/ASCII transliteration
(Condition B) — so that script-level bias can be separated from
lexical/phonological bias.

---

## 2. Data Source — `names_dataset`

**Library:** `names_dataset` (Python package)
**Underlying data:** Derived from Facebook social-network profile data,
aggregated by country. The library exposes a `NameDataset` object whose
`.first_names` dict maps every known first name to a country probability
distribution.

**What `.search(name)` returns:**
```python
{
  "first_name": {
    "country": {
      "United States": 0.45,
      "Brazil": 0.20,
      # only countries with non-None probability are listed
    },
    "gender": { ... }
  }
}
```

Country probabilities represent the relative frequency of the name across
Facebook users in each country. They do **not** necessarily sum to 1.0 because
entries with very small counts are stored as `None` and excluded.

**Key limitation:** The source is a social-network dataset, so it reflects the
demographics of Facebook's user base rather than true global population
distributions. Names common in countries with low Facebook penetration may be
under-represented.

---

## 3. Dataset Construction

### 3.1 Script history

| Script | Purpose |
|--------|---------|
| `scripts/database_v1.py` | First attempt — builds DataFrame, saves as Parquet. Does **not** include `strong_top_country`. |
| `scripts/database_v2.py` | **Canonical builder.** Adds `strong_top_country` boolean (True if `top_country_prob > 0.7`). Saves to `data/names_base.parquet`. |
| `scripts/parquet_to_sqlite.py` | Converts Parquet → SQLite. `full_countries_distribution` dict serialised to JSON string for SQLite. |
| `scripts/run-dataset.py` | Early exploration of `names_dataset` API; initial DataFrame schema design. |

### 3.2 Build logic (`database_v2.py`)

For every name in `nd.first_names`:
1. Call `nd.search(name)` to get the country distribution dict.
2. Skip names with no country data.
3. Extract `top_country` and `top_country_prob` as the argmax of the distribution.
4. Append a row.

Saved as `data/names_base.parquet`.

### 3.3 Dataset statistics

| Metric | Value |
|--------|-------|
| Total names | 727,545 |
| Unique `top_country` values | 105 |
| `top_country_prob` range | 0.100 – 1.000 |
| `top_country_prob` mean | ~0.663 |
| Names with `strong_top_country` (prob > 0.7) | ~339,711 (46.7 %) |

**Top 10 countries by name count:**

| Country | Names |
|---------|-------|
| Saudi Arabia | 29,251 |
| United States | 26,660 |
| Malaysia | 22,879 |
| South Africa | 21,320 |
| Colombia | 21,213 |
| Egypt | 20,884 |
| Italy | 17,215 |
| Russian Federation | 16,782 |
| Brazil | 16,775 |
| Nigeria | 16,598 |

---

## 4. Full Dataset Schema

All columns present in `data/names_base.parquet` after the complete pipeline
has been run.

### 4.1 Base columns (database_v2.py)

| Column | Type | Description |
|--------|------|-------------|
| `name` | string | First name as stored in `names_dataset` |
| `full_countries_distribution` | dict | Full country → probability mapping |
| `top_country` | string | Country with highest probability (full name, e.g. "United States") |
| `top_country_prob` | float | Probability for `top_country` |
| `strong_top_country` | bool | True if `top_country_prob > 0.7` |
| `agreement_score` | float | nationalize.io probability for `top_country`; NaN if not sampled |
| `n_models_used` | int | 1 if in nationalize.io sample, else 0 |

### 4.2 Preprocessing columns (preprocess_names.py)

| Column | Type | Description |
|--------|------|-------------|
| `name_script` | string | Unicode script family of first alphabetic character (e.g. Latin, Arabic, CJK, Cyrillic, Devanagari …) |
| `name_latin` | string | anyascii transliteration to ASCII/Latin; identical to `name` for already-Latin names |

### 4.3 Agreement scoring columns (enrich_names.py)

| Column | Type | Description |
|--------|------|-------------|
| `ethnicolr_race` | string | Top predicted ethnicity category from ethnicolr |
| `eth_EastAsian` | float | P(Asian,GreaterEastAsian,EastAsian) |
| `eth_Japanese` | float | P(Asian,GreaterEastAsian,Japanese) |
| `eth_IndianSubContinent` | float | P(Asian,IndianSubContinent) |
| `eth_African` | float | P(GreaterAfrican,Africans) |
| `eth_Muslim` | float | P(GreaterAfrican,Muslim) |
| `eth_British` | float | P(GreaterEuropean,British) |
| `eth_EastEuropean` | float | P(GreaterEuropean,EastEuropean) |
| `eth_Jewish` | float | P(GreaterEuropean,Jewish) |
| `eth_French` | float | P(GreaterEuropean,WestEuropean,French) |
| `eth_Germanic` | float | P(GreaterEuropean,WestEuropean,Germanic) |
| `eth_Hispanic` | float | P(GreaterEuropean,WestEuropean,Hispanic) |
| `eth_Italian` | float | P(GreaterEuropean,WestEuropean,Italian) |
| `eth_Nordic` | float | P(GreaterEuropean,WestEuropean,Nordic) |
| `langdetect_lang` | string | Top ISO 639-1 language code detected from name characters |
| `langdetect_prob` | float | Confidence of language detection |
| `agreement_score` | float | Probability nationalize.io assigns to `top_country` (sampled rows) |
| `n_models_used` | int | 1 for sampled rows, 0 otherwise |

### 4.4 Spell-check bias columns (spellcheck_names.py)

Two checkers × two conditions × (known + correction + in_dataset) = 12 columns.

| Column | Type | Description |
|--------|------|-------------|
| `hunspell_orig_known` | bool | hunspell en_US: original name is in dictionary |
| `hunspell_orig_correction` | str\|None | hunspell: top suggestion; None if known, no match, or ideographic script |
| `hunspell_latin_known` | bool | hunspell en_US: name_latin is in dictionary |
| `hunspell_latin_correction` | str\|None | hunspell: top suggestion for name_latin |
| `hunspell_orig_correction_in_dataset` | bool | Condition A correction matches a name in this dataset |
| `hunspell_latin_correction_in_dataset` | bool | Condition B correction matches a name in this dataset |
| `pysc_orig_known` | bool | pyspellchecker: original name recognised |
| `pysc_orig_correction` | str\|None | pyspellchecker: top correction for original |
| `pysc_latin_known` | bool | pyspellchecker: name_latin recognised |
| `pysc_latin_correction` | str\|None | pyspellchecker: top correction for name_latin |
| `pysc_orig_correction_in_dataset` | bool | Condition A correction matches a dataset name |
| `pysc_latin_correction_in_dataset` | bool | Condition B correction matches a dataset name |

---

## 5. Spell-Check Bias Research Design

### 5.1 Research question

Do English spell-check tools disproportionately flag names from non-Western or
non-Latin-script origins as misspellings? If so, is that bias primarily a
function of the script (the tool cannot process non-Latin characters) or of
the lexical/phonological origin of the name (the tool does not recognise the
name even when rendered in Latin letters)?

### 5.2 Double-run design

Each name is processed under two conditions:

- **Condition A (original):** The name as stored in the dataset, which may be
  in Arabic, CJK, Cyrillic, Latin, or any other Unicode script.
- **Condition B (transliterated):** The `name_latin` value produced by
  `anyascii`, so all names are in a form the spell-checkers were designed to
  handle.

| Comparison | What it reveals |
|------------|-----------------|
| A vs B for non-Latin scripts | Script-level bias: does the tool fail simply because of the character encoding? |
| Latin-Western vs Latin-non-Western names under B | Lexical/phonological bias: does the tool fail on unfamiliar name sounds regardless of script? |
| hunspell vs pyspellchecker | Whether bias is consistent across tools with different algorithms |

### 5.3 anyascii transliteration

**Library:** `anyascii` (Python)
**What it does:** Maps any Unicode character to its closest ASCII/Latin
equivalent using hand-crafted transliteration tables for every Unicode block.

**Behaviour by script:**
| Script | Example | anyascii output |
|--------|---------|-----------------|
| CJK | 张伟 | Zhang Wei |
| Arabic | محمد | mhmd *(abjad — vowels omitted)* |
| Cyrillic | Андрей | Andrey |
| Devanagari | देवेंद्र | Devendra |
| Hangul | 김철수 | Gimcheolsu |
| Hiragana/Katakana | 花子 | Hanako |

**Known limitation:** Arabic and Hebrew are abjad scripts (consonants only).
anyascii's output for Arabic names (e.g. "mhmd" for محمد) does not match the
standard English romanisation ("Muhammad"). Condition B may therefore understate
recognition rates for Arabic/Hebrew names specifically.

### 5.4 Ideographic script correction nulling

For names in CJK, Hangul, Hiragana, and Katakana scripts, each character
occupies a single Unicode codepoint. A 2-character Chinese name has string
length 2, causing English spell-checkers to return spurious edit-distance-1
corrections (e.g. "i", "oo"). Condition A corrections for these scripts are
set to `None` in both checkers. Condition B is unaffected as anyascii produces
longer Latin strings.

---

## 6. Tools Used

### 6.1 `names_dataset` (primary source)

Already used as the data source — not used again as an independent validator
(that would be circular).

### 6.2 `ethnicolr` (agreement scoring)

- **Purpose:** Predicts ethno-cultural category from names using a
  character-level LSTM trained on Wikipedia name–category associations.
- **Model:** `pred_wiki_name` (first name + last name; last name set to `""`).
- **Output:** 13 probability columns + `ethnicolr_race` (argmax category).
- **Non-Latin handling:** Ethnicolr strips non-ASCII characters during
  normalisation. Names in Arabic, CJK, Cyrillic etc. are skipped
  (`skipped_empty_after_normalization`) and produce NaN values on Pass 1.
  **Pass 2** re-runs ethnicolr using `name_latin` for these rows, filling in
  predictions based on the transliterated form.
- **Limitation:** Predicts broad regional categories, not specific countries.
  Useful as a regional corroboration signal and for data-science analysis of
  the dataset structure.

### 6.3 `langdetect` (agreement scoring — weak signal)

- **Purpose:** Detects the script/language of a text string from character
  n-gram statistics.
- **Workaround:** Each name is repeated 5× to inflate the character sequence
  (e.g. `"Pierre Pierre Pierre Pierre Pierre "`). `DetectorFactory.seed = 0`
  for reproducibility.
- **Known limitation:** Results are noisy for short strings. `Pierre → fr` ✓,
  `Bjorn → no` ✓, but `Fatima → it` ✗, `Akira → lt` ✗. Included as a
  supporting signal to see whether it provides aggregate corroboration, not as
  a standalone predictor.
- **Runtime:** ~2–3 hours for 727,545 names via pandas `.apply()`. The main
  bottleneck in the pipeline.

### 6.4 `nationalize.io` (agreement scoring — stratified sample)

- **Purpose:** Predicts nationality from first names via a free REST API.
  Returns ISO alpha-2 codes with probabilities.
- **Rate limits:** 100 requests/day (no key); 1,000/day (free API key).
- **Why sampled:** 727,545 names at 1,000/day = ~2 years. A stratified sample
  of 1,000 names (250 per `top_country_prob` quartile bin) characterises
  agreement across the full confidence range.
- **Agreement score formula:** The probability nationalize.io assigns to the
  ISO alpha-2 equivalent of `top_country`. If the country is absent from the
  response, score = 0.0. Range: [0, 1].
- **Country code mapping:** nationalize.io returns ISO alpha-2 (e.g. "US");
  the dataset stores full country names (e.g. "United States"). Mapping via
  `pycountry`. All 105 unique `top_country` values resolved successfully.

### 6.5 `hunspell` via `pyenchant` (spell-check — primary)

- **Purpose:** Industry-standard spell-checker. The engine behind LibreOffice,
  Firefox, macOS system spell check, and Chrome. Primary tool for the bias
  analysis because it represents real-world user experience.
- **Dictionary:** `en_US`.
- **API:** `enchant.Dict.check(word)` → bool; `enchant.Dict.suggest(word)` →
  list of suggestions (top suggestion taken).
- **System dependency:** Requires `libenchant-2-dev` + `hunspell-en-us` (Linux)
  or the bundled Enchant library (Windows/macOS via pyenchant wheel).

### 6.6 `pyspellchecker` (spell-check — secondary baseline)

- **Purpose:** Simpler spell-checker using a pre-built English word-frequency
  dictionary and Damerau-Levenshtein edit-distance candidate generation (Norvig
  approach). Algorithmically distinct from hunspell; included to verify that
  bias findings are not tool-specific.
- **API:** `SpellChecker.known(word_list)` → set; `SpellChecker.correction(word)`
  → top correction.
- **Note:** There are two PyPI packages named "spellchecker" (the old,
  unmaintained one) and "pyspellchecker" (the correct one). Both install under
  the same import `from spellchecker import SpellChecker`. Install with
  `pip install pyspellchecker`.

### 6.7 Tools investigated and rejected

| Tool | Reason not used |
|------|----------------|
| `name2nat` | PyTorch version incompatibility — GRU `_flat_weights` attribute renamed in PyTorch ≥ 1.8. Fails on Python 3.11 and 3.13. Would require a Python 3.6 legacy environment. |
| NamSor API | Free tier is 500 credits/month (~16/day) — too limited for meaningful sampling. |
| `langid` / `polyglot` | Same language-detection limitations as `langdetect`, no benefit over it. |
| Web scraping (forebears.io etc.) | Against terms of service; fragile. |

---

## 7. Pipeline Architecture

### 7.1 Script inventory

| Script | Stage | Description |
|--------|-------|-------------|
| `scripts/database_v2.py` | 1 | Builds base parquet from `names_dataset`. |
| `scripts/preprocess_names.py` | 2 | Adds `name_script` (Unicode script detection) and `name_latin` (anyascii transliteration). Must run before steps 3 and 4. |
| `scripts/enrich_names.py` | 3 | Agreement scoring: ethnicolr (2 passes), langdetect, nationalize.io sample. Saves checkpoints after each step. |
| `scripts/spellcheck_names.py` | 4 | Spell-check bias analysis: hunspell + pyspellchecker under Conditions A and B. Saves checkpoint after hunspell. |
| `scripts/parquet_to_sqlite.py` | — | Optional: converts final parquet to SQLite for SQL querying. |
| `scripts/run-dataset.py` | — | Early exploration; not part of production pipeline. |
| `scripts/playground.py` | — | Tool exploration (ethnicolr, langdetect samples). |
| `scripts/playground_name2nat.py` | — | name2nat testing; confirmed broken. |

### 7.2 Step order and dependencies

```
database_v2.py          →  data/names_base.parquet (base columns only)
preprocess_names.py     →  + name_script, name_latin
enrich_names.py         →  + eth_*, langdetect_*, agreement_score
                              (uses name_latin for ethnicolr pass 2)
spellcheck_names.py     →  + hunspell_*, pysc_*
                              (reads name_latin; fails if preprocess not run)
```

Each step reads the parquet, adds its columns, and writes it back in-place.

### 7.3 Checkpointing

`enrich_names.py` saves the parquet to disk after each major step:

| Checkpoint | Columns saved |
|-----------|---------------|
| After ethnicolr | eth_* + ethnicolr_race |
| After langdetect | + langdetect_lang/prob |
| After nationalize.io | + agreement_score (final save) |

`spellcheck_names.py` saves after hunspell completes, before running
pyspellchecker. If any later step crashes, the last checkpoint is already on
disk.

### 7.4 Environment and secrets

| Variable | Where stored |
|----------|-------------|
| `NATIONALIZE_KEY` | `.env` file locally (gitignored); GitHub Actions repository secret for CI runs |

Load with `python-dotenv` (`load_dotenv()` at script start). On GitHub Actions
the secret is injected directly as an environment variable; `.env` is not
present and `load_dotenv()` is a no-op.

---

## 8. GitHub Actions Pipeline

### 8.1 Workflow file

`.github/workflows/pipeline.yml` — triggered manually (`workflow_dispatch`)
from the Actions tab.

### 8.2 Step sequence

| Step | Script | Notes |
|------|--------|-------|
| 1 | `database_v2.py` | Rebuilds parquet from scratch — keeps large data files out of the repository |
| 2 | `preprocess_names.py` | Script detection + transliteration |
| 3 | `enrich_names.py` | Uses `NATIONALIZE_KEY` secret |
| 4 | `spellcheck_names.py` | Requires `libenchant-2-dev` + `hunspell-en-us` system packages |

### 8.3 Artifacts

Two artifacts are uploaded after each run with `if: always()` (so partial
results are preserved even if a step fails):
- `names-enriched-<run_id>` — `data/names_base.parquet`
- `logs-<run_id>` — all `.log` and `_report.md` files

**To access:** GitHub repo → Actions tab → click the run → scroll to Artifacts
section → download as zip.

### 8.4 Runtime estimate

| Step | Estimated time |
|------|---------------|
| database_v2.py | ~20–30 min |
| preprocess_names.py | ~5–10 min |
| enrich_names.py (ethnicolr × 2) | ~3 min |
| enrich_names.py (langdetect) | ~2–3 hours *(main bottleneck)* |
| enrich_names.py (nationalize.io) | ~10 min |
| spellcheck_names.py (hunspell) | ~15–30 min |
| spellcheck_names.py (pyspellchecker) | ~30–90 min |
| **Total** | **~4–6 hours** |

The 6-hour `timeout-minutes` cap in the workflow accommodates the full run.
The pipeline is designed to be triggered once (or re-run if a step fails) with
all processing on GitHub's infrastructure.

---

## 9. Key Decisions Log

| Decision | Rationale |
|----------|-----------|
| Parquet as primary format | Compact, typed, fast with pandas. SQLite generated optionally for ad-hoc SQL querying. |
| `strong_top_country` threshold of 0.7 | Round-number boundary separating names with a clear dominant origin from ambiguous ones. 46.7% of names exceed it. |
| Use `pred_wiki_name` with empty last name | Dataset contains first names only. Wiki model chosen over Florida voter-registration model for broader international coverage. |
| Repeat name 5× for langdetect | Below ~20 characters langdetect frequently raises `LangDetectException`. Repetition inflates the character sequence without changing n-gram composition. |
| Stratified sample for nationalize.io | Equal sampling from four probability quartile bins ensures agreement score is computed over the full confidence range, not just easy high-confidence names. |
| Agreement score = nationalize.io probability | A continuous score (not binary match/no-match) preserves the strength of agreement. |
| anyascii for transliteration | Handles all Unicode scripts via hand-crafted tables. Produces good output for CJK (pinyin), Cyrillic, Devanagari. Main weakness: Arabic/Hebrew lose vowels (abjad scripts). |
| Double-run spell-check design | Separates script-level bias (Condition A vs B) from lexical/phonological bias (Latin Western vs Latin non-Western under B). |
| hunspell as primary spell-checker | Industry standard — powers LibreOffice, Firefox, macOS, Chrome. Bias findings here have direct real-world significance. |
| pyspellchecker as secondary baseline | Algorithmically distinct (frequency dict + edit distance vs affix rules). Corroborates or contrasts hunspell findings. |
| Null CJK/Hangul/Hiragana/Katakana Condition A corrections | Single-codepoint characters produce spurious edit-distance-1 matches with short English words (e.g. "i"). These corrections are meaningless and would distort the analysis. |
| Ethnicolr second pass using name_latin | ~18.3% of names are non-Latin script and are skipped by ethnicolr on Pass 1. Running Pass 2 with anyascii transliteration fills these NaN values, giving coverage across all scripts. |
| Checkpoint saves within long scripts | enrich_names.py and spellcheck_names.py save the parquet after each major step so that a crash in a later step does not lose hours of completed work. |
| Secrets in .env locally / GitHub Secrets in CI | .env is gitignored. The same environment variable (`NATIONALIZE_KEY`) is used in both contexts; `load_dotenv()` is a no-op when the variable is already in the environment. |
| name2nat rejected | Confirmed broken on Python 3.11 and 3.13 due to PyTorch GRU serialisation format change (`_flat_weights` → `_all_weights`). Would require a Python 3.6 legacy environment. |

---

## 10. File Structure

```
HonsProject/
├── .env                         # Local secrets (gitignored)
├── .github/
│   └── workflows/
│       └── pipeline.yml         # GitHub Actions pipeline (manual trigger)
├── .gitignore                   # Excludes data/, .env, logs/, __pycache__/
├── requirements.txt             # All Python dependencies
├── PROJECT_NOTES.md             # This file
├── data/                        # Gitignored — generated by pipeline
│   ├── names_base.parquet       # Primary dataset (all pipeline columns)
│   ├── names.db                 # SQLite copy v1 (optional)
│   └── names2.db                # SQLite copy v2 (optional)
├── logs/                        # Gitignored — generated by scripts
│   ├── spellcheck_<ts>.log
│   ├── spellcheck_<ts>_report.md
│   └── (enrich logs to be added)
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   └── 02_explore_names2.ipynb
└── scripts/
    ├── database_v2.py           # Pipeline step 1: build base dataset
    ├── preprocess_names.py      # Pipeline step 2: name_script + name_latin
    ├── enrich_names.py          # Pipeline step 3: agreement scoring
    ├── spellcheck_names.py      # Pipeline step 4: spell-check bias analysis
    ├── parquet_to_sqlite.py     # Optional: convert parquet → SQLite
    ├── run-dataset.py           # Early exploration (not production)
    ├── playground.py            # Tool exploration (not production)
    └── playground_name2nat.py   # name2nat testing (not production)
```
