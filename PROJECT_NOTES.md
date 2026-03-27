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
3. Skip names where every space-separated part is a single character (e.g. "A A" is removed, but "A Abdul" is kept). These are data artefacts from the source dataset rather than real names.
4. Extract `top_country` and `top_country_prob` as the argmax of the distribution.
5. Append a row.

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
| `ethnicolr_race` | string | Top predicted ethnicity category from ethnicolr (full hierarchical label, e.g. "GreaterEuropean,British") |
| `ethnicolr_prob` | float | Probability assigned to `ethnicolr_race` |
| `eth_distribution` | dict | All 13 ethnicity probabilities as a single dict, e.g. `{"British": 0.40, "EastAsian": 0.09, ...}`. Short category names: EastAsian, Japanese, IndianSubContinent, African, Muslim, British, EastEuropean, Jewish, French, Germanic, Hispanic, Italian, Nordic. |
| `langdetect_lang` | string | Top ISO 639-1 language code detected from name characters |
| `langdetect_lang_name` | string | Full English language name for `langdetect_lang` (e.g. "Arabic", "Hungarian") |
| `langdetect_prob` | float | Confidence of language detection |
| `top_country_langs` | list | Official language codes (ISO 639-1) for `top_country`, from countryinfo |
| `country_lang_comp` | bool | True if `langdetect_lang` is in `top_country_langs` — rough consistency check |
| `agreement_score` | float | Probability nationalize.io assigns to `top_country` (sampled rows only; NaN otherwise) |
| `n_models_used` | int | Count of models that successfully ran on this name (ethnicolr, langdetect each +1; nationalize.io +1 for sampled rows) |

**Note on `eth_distribution`:** The 13 individual `eth_*` probability columns that ethnicolr originally produces are consolidated into this single dict column. This keeps the schema clean while retaining all the probability information. Values are rounded to 4 decimal places.

### 4.4 Spell-check known/unknown columns (spellcheck_names.py → `names_results_base.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| `hunspell_orig_known` | bool | hunspell en_US: original name is in dictionary |
| `hunspell_latin_known` | bool | hunspell en_US: name_latin is in dictionary |
| `pysc_orig_known` | bool | pyspellchecker: original name recognised |
| `pysc_latin_known` | bool | pyspellchecker: name_latin recognised |

### 4.5 Correction columns (corrections_names.py → `advanced_results_base.parquet`)

Three checkers (hunspell, symspell, lt) × two conditions (orig, latin) = 6 correction string columns + 6 match detail columns.

Correction string columns hold the top suggestion returned by the tool, or None if the name was known / no suggestion / ideographic script (Condition A only). Match detail columns hold None if there was no correction or the correction does not match any name in the dataset, otherwise a dict of that matched name's characteristics.

| Column | Type | Description |
|--------|------|-------------|
| `hunspell_orig_correction` | str\|None | top hunspell suggestion for original name |
| `hunspell_latin_correction` | str\|None | top hunspell suggestion for name_latin |
| `hunspell_orig_correction_match` | dict\|None | match details if correction is a dataset name |
| `hunspell_latin_correction_match` | dict\|None | match details if correction is a dataset name |
| `symspell_orig_correction` | str\|None | top symspell suggestion for original name |
| `symspell_latin_correction` | str\|None | top symspell suggestion for name_latin |
| `symspell_orig_correction_match` | dict\|None | match details if correction is a dataset name |
| `symspell_latin_correction_match` | dict\|None | match details if correction is a dataset name |
| `lt_orig_correction` | str\|None | top languagetool suggestion for original name |
| `lt_latin_correction` | str\|None | top languagetool suggestion for name_latin |
| `lt_orig_correction_match` | dict\|None | match details if correction is a dataset name |
| `lt_latin_correction_match` | dict\|None | match details if correction is a dataset name |

Match dict fields: `name`, `top_country`, `name_script`, `ethnicolr_race`, `langdetect_lang`, `matched_via_latin`. The `matched_via_latin` bool is True when the correction matched via the name_latin transliteration of a dataset entry rather than its original form. Lookup is case-insensitive throughout. Name-form matches take priority over latin-form matches for the same key.

---

## 5. Spell-Check Bias Research Design

### 5.1 Research question

Do English spell-check tools disproportionately flag names from non-Western or
non-Latin-script origins as misspellings? If so, is that bias primarily a
function of the script (the tool cannot process non-Latin characters) or of
the lexical/phonological origin of the name (the tool does not recognise the
name even when rendered in Latin letters)?

### 5.1.1 Dataset cleaning for the spell-check analysis

Before running the spell-check analysis, a cleaning pass filters the dataset
to names that are meaningful inputs for a spell-checker:

**Kept:**
- Single-word names (no spaces) — multi-word entries like `"A Abdul"` are composites, not single given names
- Names containing only Unicode letters (`L*` category) and combining marks (`M*` category)
- Names with accented Latin characters (e.g. `Nguyễn`, `José`) — these are legitimate names and the accent is meaningful
- Names in non-Latin scripts (Arabic, Cyrillic, Devanagari, CJK etc.) — kept for the Condition A/B comparison

**Filtered out:**
- Names containing digits (`0–9`)
- Names containing noise characters: `*`, `.`, `,`, `_`, `+`, `=`, `@`, `#`, etc.
- Hyphens and apostrophes are borderline — still under consideration (e.g. `Mary-Jane`, `O'Brien` are legitimate names; but `*-` style entries are artefacts)

**Rationale:** The core research question is about phonological/lexical bias, not about how spell-checkers handle punctuation or digits. Noise characters would make it impossible to distinguish "spell-checker doesn't recognise this name" from "spell-checker doesn't accept strings with punctuation in them". Cleaning isolates the variable of interest.

**Note on combining marks:** Initial filter implementation only allowed Unicode letter categories (`L*`), which incorrectly excluded Devanagari names like `देवेंद्र` (contains `ं` anusvara and `्र` virama, both category `M`). Fixed to allow both `L*` and `M*` categories.

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

### 5.4 Sentence-context experiment (investigated, rejected)

Before committing to the full pipeline, a sandbox test (`scripts/sandbox_sentence_check.py`) investigated whether checking a name in a sentence context would yield different results from checking it standalone. Four formats were tested:

| Format | Example |
|--------|---------|
| Standalone | `Fatima` |
| Sentence-end | `My name is Fatima` |
| Sentence-start | `Fatima is my name` |
| Sentence-middle | `My name is Fatima and that is a fact` |

**Finding — Approach A (whole sentence string passed to hunspell):** Hunspell treats the entire string as one word and applies edit-distance substitutions at arbitrary positions across the full string (e.g. `"My name is Fatima"` → `['My neime is Fatima', 'My neyme is Fatima']`). The suggestions are meaningless noise and bear no relation to name correction. Not usable.

**Finding — Approach B (sentence split into tokens, each checked separately):** The result for the name token is identical regardless of whether it appears standalone, at the start, middle, or end of a sentence. Hunspell is purely word-level with no context awareness. Sentence position adds nothing.

**Conclusion:** The sentence-context design was abandoned. Hunspell is word-level; position is irrelevant. Adding sentence variants would triple compute time with no additional information. The double-run design (original vs. transliterated) remains the primary comparison axis.

### 5.5 Ideographic script correction nulling

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
| `scripts/database_v2.py` | 1 | Builds base parquet from `names_dataset`. Outputs `names_base.parquet`. |
| `scripts/preprocess_names.py` | 2 | Adds `name_script` and `name_latin`. In-place on `names_base.parquet`. Must run before steps 3 and 4. |
| `scripts/enrich_names.py` | 3 | Agreement scoring: ethnicolr (2 passes), langdetect, nationalize.io sample. Reads `names_base.parquet`, saves `names_results_base.parquet`. |
| `scripts/spellcheck_names.py` | 4 | Spell-check known/unknown analysis. Reads and updates `names_results_base.parquet` in-place. |
| `scripts/corrections_names.py` | 5 | Computes correction suggestions. Reads `names_results_base.parquet`, outputs `advanced_results_base.parquet`. Runs in parallel chunks on GitHub Actions. |
| `scripts/parquet_to_sqlite.py` | — | Optional: converts any parquet to SQLite for SQL querying. Dict/array columns serialised to JSON strings. |
| `scripts/run-dataset.py` | — | Early exploration; not part of production pipeline. |
| `scripts/playground.py` | — | Tool exploration (ethnicolr, langdetect samples). |
| `scripts/playground_name2nat.py` | — | name2nat testing; confirmed broken. |

### 7.2 Step order and dependencies

```
database_v2.py          →  data/names_base.parquet
                              (base columns + name_script + name_latin)
enrich_names.py         →  data/names_results_base.parquet
                              (+ ethnicolr_race, ethnicolr_prob, eth_distribution,
                                 langdetect_*, top_country_langs, country_lang_comp,
                                 agreement_score)
                              [uses name_latin for ethnicolr pass 2]
spellcheck_names.py     →  data/names_results_base.parquet  [in-place]
                              (+ hunspell_*_known, pysc_*_known)
corrections_names.py    →  data/advanced_results_base.parquet
                              (+ hunspell_*_correction, pysc_*_correction, *_in_dataset)
```

Steps 1–4 run sequentially in one GitHub Actions job (`pipeline.yml`). The corrections step (step 5) runs as a separate workflow triggered manually after the pipeline completes.

**Parquet file naming:**
- `names_base.parquet` — output of database + preprocess steps
- `names_results_base.parquet` — output of enrich + spellcheck steps (primary analysis file)
- `advanced_results_base.parquet` — output of corrections step (adds suggestion columns)

### 7.3 Checkpointing

`enrich_names.py` saves `names_results_base.parquet` after each major step:

| Checkpoint | Columns saved at that point |
|-----------|------------------------------|
| After ethnicolr | ethnicolr_race, ethnicolr_prob, eth_distribution |
| After langdetect | + langdetect_lang, langdetect_lang_name, langdetect_prob |
| After nationalize.io | + agreement_score, top_country_langs, country_lang_comp (final save) |

`spellcheck_names.py` saves `names_results_base.parquet` after hunspell
completes, before running pyspellchecker. If any later step crashes, the
last checkpoint is already on disk and will be captured by the GitHub Actions
artifact upload (`if: always()`).

### 7.4 Environment and secrets

| Variable | Where stored |
|----------|-------------|
| `NATIONALIZE_KEY` | `.env` file locally (gitignored); GitHub Actions repository secret for CI runs |

Load with `python-dotenv` (`load_dotenv()` at script start). On GitHub Actions
the secret is injected directly as an environment variable; `.env` is not
present and `load_dotenv()` is a no-op.

---

## 8. GitHub Actions Pipeline

### 8.1 Workflow files

Three separate workflow files, all triggered manually (`workflow_dispatch`):

| File | Purpose | Trigger inputs |
|------|---------|----------------|
| `pipeline.yml` | Steps 1–4: build, preprocess, enrich, spellcheck | none |
| `corrections_hunspell.yml` | Step 5a: hunspell corrections in 2 parallel chunks | `source_run_id` (pipeline run to download parquet from) |
| `corrections_pysc.yml` | Step 5b: pyspellchecker corrections + merge | `source_run_id` (pipeline run), `hunspell_run_id` (hunspell workflow run) |

The corrections are split across two separate workflows (rather than two jobs in one file) because each step gets its own 6-hour budget. When they were in one workflow the total wall time exceeded the 6-hour hard cap.

### 8.2 pipeline.yml — step sequence

| Step | Script | Notes |
|------|--------|-------|
| 1 | `database_v2.py` | Rebuilds parquet from scratch — keeps large data files out of the repository |
| 2 | `preprocess_names.py` | Script detection + transliteration |
| 3 | `enrich_names.py` | Uses `NATIONALIZE_KEY` secret |
| 4 | `spellcheck_names.py` | Requires `libenchant-2-dev` + `hunspell-en-us` system packages |

**Artifacts uploaded** (with `if: always()` so partial results are preserved):
- `names-enriched-<run_id>` — `data/names_results_base.parquet`
- `logs-<run_id>` — all `.log` and `_report.md` files

### 8.3 corrections_hunspell.yml — parallel chunk strategy

hunspell's `suggest()` is ~0.025 s/word. With ~786,000 unique unknowns across both conditions, serial processing takes ~5.5 hours — over the 6-hour limit. Solution: split unknowns alphabetically into 2 chunks using a matrix strategy, run both in parallel (~3–3.5 hours each).

Each chunk job:
1. Downloads `names_results_base.parquet` from the specified `source_run_id`
2. Sorts all unknowns alphabetically and takes its slice
3. Runs `corrections_names.py --mode hunspell-chunk --chunk N --total-chunks 2`
4. Uploads `data/hunspell_corrections_chunk_N.json` as `hunspell-chunk-N-<run_id>` (7-day retention)

Cross-run artifact downloads require `permissions: actions: read`.

### 8.4 corrections_pysc.yml — merge and finalise

1. Downloads `names_results_base.parquet` (from `source_run_id`)
2. Downloads both `hunspell_corrections_chunk_0` and `hunspell_corrections_chunk_1` (from `hunspell_run_id`)
3. Runs `corrections_names.py --mode pysc --total-chunks 2`
   - Merges both JSON chunk files into one correction map
   - Applies hunspell corrections to the dataframe
   - Runs pyspellchecker corrections
   - Saves `data/advanced_results_base.parquet`
4. Uploads as `names-advanced-<run_id>` (90-day retention)

### 8.5 Runtime estimates

| Step | Estimated time |
|------|---------------|
| database_v2.py | ~20–30 min |
| preprocess_names.py | ~5–10 min |
| enrich_names.py (ethnicolr × 2 passes) | ~3 min |
| enrich_names.py (langdetect) | ~2–3 hours *(main bottleneck)* |
| enrich_names.py (nationalize.io) | ~10 min |
| spellcheck_names.py (hunspell known/unknown) | ~15–30 min |
| spellcheck_names.py (pyspellchecker known/unknown) | ~30–90 min |
| **pipeline.yml total** | **~4–6 hours** |
| corrections_hunspell.yml (each of 2 parallel chunks) | ~3–3.5 hours |
| corrections_pysc.yml | ~2+ hours |

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
| eth_distribution replaces 13 eth_* columns | Storing 13 individual float columns is verbose and wasteful for a dataset used primarily via pandas/SQL. A single dict column captures all the information in a more compact schema. Values are rounded to 4 decimal places. |
| Hunspell corrections split into 2 parallel chunks | ~786k unique unknowns × 0.025 s = ~5.5 hours serial. Splitting alphabetically into 2 chunks of ~393k each yields ~3–3.5 hours per chunk, within the 6-hour GitHub Actions job limit. |
| Separate corrections workflows (not two jobs in one file) | Even after parallelising hunspell, the pysc corrections step alone takes 2+ hours. If both lived in one workflow file, the total wall time (hunspell wait + pysc) would exceed 6 hours. Separate workflows each get their own 6-hour budget. |
| Filter single-character-part names | Names like "A A" where every space-separated token is ≤ 1 character are artefacts from the names_dataset source, not real names. Removing them avoids polluting spell-check results with trivially unrecognisable tokens. |
| *_correction_in_dataset flag | Checking whether the suggested correction is itself a name in the dataset surfaces cases where the spell-checker nudges a non-Western name towards a more Western/common alternative — a potentially interesting secondary bias signal. |
| Sentence context investigated and rejected | Tested 4 formats (standalone, sentence-start, sentence-mid, sentence-end). Approach A (whole string to hunspell) produces garbage edit-distance substitutions across the full string. Approach B (token-level) gives identical results in all positions — hunspell has no context awareness. Sentence variants would triple compute time for zero additional information. |
| Clean filter allows Unicode combining marks (M*) | Initial filter restricted to letter categories (L*) only, which incorrectly excluded valid names in scripts that use combining diacritics (Devanagari anusvara, Arabic harakat). Fixed to allow L* and M* categories together. |
| Dataset cleaning to single-word, noise-free names | For the spell-check analysis, multi-word entries and names with digits or noise characters (*, ., etc.) are filtered out. This isolates the phonological/lexical bias signal from confounds introduced by punctuation and non-name tokens. Accented Latin and non-Latin scripts are kept. |

---

## 10. File Structure

```
HonsProject/
├── .env                              # Local secrets (gitignored)
├── .github/
│   └── workflows/
│       ├── pipeline.yml              # Steps 1–4: build → enrich → spellcheck
│       ├── corrections_hunspell.yml  # Step 5a: parallel hunspell chunks
│       └── corrections_pysc.yml      # Step 5b: pysc corrections + merge
├── .gitignore                        # Excludes data/, .env, logs/, __pycache__/
├── requirements.txt                  # All Python dependencies
├── PROJECT_NOTES.md                  # This file
├── data/                             # Gitignored — generated by pipeline
│   ├── names_base.parquet            # After steps 1–2 (base + script + latin)
│   ├── names_results_base.parquet    # After steps 3–4 (enrich + spellcheck)
│   ├── advanced_results_base.parquet # After step 5 (corrections)
│   └── names.db                      # SQLite copy (optional, from parquet_to_sqlite.py)
├── logs/                             # Gitignored — generated by scripts
│   ├── spellcheck_<ts>.log
│   ├── spellcheck_<ts>_report.md
│   └── corrections_<mode>_<ts>.log
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   ├── 02_explore_names2.ipynb
│   └── 03_explore_names_results.ipynb
└── scripts/
    ├── database_v2.py                # Step 1: build base dataset
    ├── preprocess_names.py           # Step 2: name_script + name_latin
    ├── enrich_names.py               # Step 3: ethnicolr + langdetect + nationalize.io
    ├── spellcheck_names.py           # Step 4: known/unknown spell-check
    ├── corrections_names.py          # Step 5: correction suggestions (chunked)
    ├── parquet_to_sqlite.py          # Optional: convert parquet → SQLite
    ├── run-dataset.py                # Early exploration (not production)
    ├── playground.py                 # Tool exploration (not production)
    └── playground_name2nat.py        # name2nat testing (not production)
```
