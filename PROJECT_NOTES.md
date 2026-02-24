# Project Notes — Name Origin Dataset

A running record of decisions, methodology, and findings for reference when writing up.

---

## 1. Project Overview

The goal is to build a dataset of first names enriched with country-of-origin
predictions and a measure of how well those predictions are corroborated by
independent tools (an *agreement score*). The dataset is intended to support
data-science analysis of name origins at scale.

---

## 2. Data Source — `names_dataset`

**Library:** `names_dataset` (Python package)
**Underlying data:** Derived from Facebook social-network profile data, aggregated
by country. The library exposes a `NameDataset` object whose `.first_names` dict
maps every known first name to a country probability distribution.

**What `.search(name)` returns:**
```python
{
  "first_name": {
    "country": {
      "United States": 0.45,
      "Brazil": 0.20,
      ...          # only countries with non-None probability are listed
    },
    "gender": {...}
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
| `scripts/database_v1.py` | First attempt — iterates `nd.first_names`, builds DataFrame, saves as Parquet. Does **not** include `strong_top_country`. |
| `scripts/database_v2.py` | Revised version — identical logic but adds the `strong_top_country` boolean column (True if `top_country_prob > 0.7`) and saves to `data/names_base.parquet`. This is the canonical source file. |
| `scripts/parquet_to_sqlite.py` | Converts the Parquet to SQLite (`data/names.db` / `data/names2.db`). The `full_countries_distribution` dict is serialised to a JSON string for SQLite storage, since SQLite has no native map type. |
| `scripts/run-dataset.py` | Early exploration script — tests `nd.search()` and `NameWrapper.describe` output before the full build. Also contains the initial DataFrame schema design with comments about intended future columns. |

### 3.2 Build logic (database_v2.py)

For every name in `nd.first_names`:
1. Call `nd.search(name)` to get the country distribution dict.
2. Skip names with no country data.
3. Extract `top_country` and `top_country_prob` as the argmax of the distribution.
4. Append a row to the list.

The final DataFrame is saved as `data/names_base.parquet`.

### 3.3 Dataset statistics (as built)

| Metric | Value |
|--------|-------|
| Total rows (names) | 727,545 |
| Unique `top_country` values | 105 |
| `top_country_prob` range | 0.100 – 1.000 |
| `top_country_prob` mean | 0.663 |
| Names with `strong_top_country` (prob > 0.7) | 339,711 (46.7 %) |

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

## 4. Database Schema

### Parquet / SQLite `names` table

| Column | Type | Description |
|--------|------|-------------|
| `name` | string | First name (as returned by `names_dataset`) |
| `full_countries_distribution` | dict / JSON string | Full country → probability mapping from `names_dataset`. `None` values indicate the name occurs in that country but the count is too small to give a reliable probability. |
| `top_country` | string | Country with the highest probability (full ISO 3166-1 country name, e.g. "United States") |
| `top_country_prob` | float | Probability value for `top_country` |
| `strong_top_country` | bool | True if `top_country_prob > 0.7` — indicates a clear dominant origin |
| `agreement_score` | float | Probability that an independent tool (nationalize.io) assigns to `top_country` for this name. NaN for names not in the sample. |
| `n_models_used` | int | Number of external tools queried for this name's agreement score (0 or 1) |

**Planned enrichment columns** (added by `scripts/enrich_names.py`):

| Column | Source | Scope |
|--------|--------|-------|
| `ethnicolr_race` | ethnicolr | All names |
| `eth_EastAsian` … `eth_Nordic` (13 cols) | ethnicolr | All names |
| `langdetect_lang` | langdetect | All names |
| `langdetect_prob` | langdetect | All names |
| `agreement_score` | nationalize.io | Stratified sample of 1,000 |
| `n_models_used` | — | Stratified sample of 1,000 |

---

## 5. Tools Investigated for External Validation

### 5.1 `names_dataset` (primary source)

Already used as the source data — cannot be used again as an independent validator.

### 5.2 `name2nat`

- **Purpose:** Predicts nationality from full names using a GRU-based neural network
  (built on the Flair NLP framework).
- **Status: Unusable.** The saved model was serialised with an old version of
  PyTorch that stored GRU weights under the attribute `_flat_weights`. Newer
  PyTorch (≥ 1.8) renamed this to `_all_weights`. The model therefore fails to
  load on Python 3.11 and 3.13 with:
  `AttributeError: 'GRU' object has no attribute '_flat_weights'`
- **Conclusion:** Requires a PyTorch ~1.4 environment (Python 3.6–3.7). Not
  viable without maintaining a separate legacy environment.

### 5.3 `ethnicolr`

- **Purpose:** Predicts ethnicity/race from names using a character-level LSTM
  trained on Wikipedia name–category associations.
- **Model used:** `pred_wiki_name` (takes first name + last name; last name set
  to empty string since only first names are available).
- **Status: Adopted for full dataset.**
- **Output:** 13 hierarchical probability columns + a top-category `race` string.
- **Limitation:** Predicts broad ethno-cultural categories, not specific countries.
  "GreaterAfrican,Muslim" spans dozens of countries; "GreaterEuropean,British"
  conflates England, Australia, Canada, etc. Useful as a regional corroboration
  signal but cannot replace country-level predictions.
- **Why included despite limitation:** It runs entirely locally with no rate
  limit, making it feasible for all 727,545 names. The 13 probability dimensions
  add orthogonal signal for data-science analysis (e.g. correlation between
  `eth_French` and names with `top_country = France`).

### 5.4 `langdetect`

- **Purpose:** Detects the language of a text string from character n-gram
  statistics.
- **Status: Adopted for full dataset (as a weak supporting signal).**
- **Workaround applied:** `langdetect` was designed for paragraphs. Individual
  names are too short for reliable detection. Each name is repeated 5 times
  (e.g. `"Pierre Pierre Pierre Pierre Pierre "`) before detection to increase
  the character n-gram signal.
- **Seed fixed at 0** for reproducibility (the library is non-deterministic by
  default).
- **Limitation:** Results are noisy. `Pierre → fr` ✓, `Bjorn → no` ✓, but
  `Fatima → it` ✗ (should be Arabic-origin), `Akira → lt` ✗ (should be
  Japanese-origin). Not used as a standalone predictor — included to see whether
  it provides any corroborating signal in aggregate.

### 5.5 `nationalize.io`

- **Purpose:** Predicts nationality from first names via a free REST API. Returns
  a ranked list of ISO 3166-1 alpha-2 codes with probabilities. Structurally
  identical to the dataset's own `full_countries_distribution`.
- **Status: Adopted for stratified sample (agreement score).**
- **API:** `GET https://api.nationalize.io?name[]=<name>&name[]=<name>&…`
- **Rate limits:** 100 requests/day without an API key; 1,000/day with a free
  key (register at nationalize.io).
- **Why sampled rather than full dataset:** 727,545 individual names at 1,000/day
  would take ~2 years. A stratified sample of 1,000 names (250 from each
  `top_country_prob` quartile) is sufficient to characterise agreement across
  the full confidence range.
- **Country code mapping:** `nationalize.io` returns ISO alpha-2 codes (e.g.
  "US"). The dataset stores full country names (e.g. "United States"). Mapping
  is done via the `pycountry` library. All 105 unique `top_country` values in
  the dataset resolved successfully.
- **Agreement score formula:** For a sampled name, the score equals the
  probability that `nationalize.io` assigns to the ISO alpha-2 equivalent of
  `top_country`. If the country does not appear in the response, the score is
  0.0. The score lies in [0, 1].

### 5.6 Tools considered and rejected

| Tool | Reason not used |
|------|----------------|
| `name2nat` | PyTorch version incompatibility — see §5.2 |
| NamSor API | Free tier is 500 credits/month (~16/day), too limited for meaningful sampling |
| `langid` / `polyglot` | Language detection proxies — same limitations as `langdetect` |
| Web scraping (forebears.io etc.) | Against terms of service; fragile |

---

## 6. Script Inventory

| Script | Description |
|--------|-------------|
| `scripts/run-dataset.py` | Early exploration of `names_dataset` API; defines initial DataFrame schema with design comments. |
| `scripts/database_v1.py` | First full dataset build — generates Parquet without `strong_top_country`. |
| `scripts/database_v2.py` | **Canonical dataset builder.** Adds `strong_top_country`, saves to `data/names_base.parquet`. |
| `scripts/parquet_to_sqlite.py` | Converts `data/names_base.parquet` → `data/names2.db` (SQLite). Serialises `full_countries_distribution` dict to JSON string. |
| `scripts/playground.py` | Tool exploration — tests `ethnicolr.pred_wiki_name` and `langdetect` on a small set of sample names. |
| `scripts/playground_name2nat.py` | Tool exploration — tests `name2nat` on known-nationality politicians. Confirmed that `name2nat` works best with full names (first + last). |
| `scripts/enrich_names.py` | **Main enrichment script.** Runs ethnicolr and langdetect on all 727,545 names, queries nationalize.io for a stratified sample, and writes a detailed markdown report for every run. |

---

## 7. Enrichment Methodology (enrich_names.py)

### Step 1 — ethnicolr (all names)

- Function: `ethnicolr.pred_wiki_name(df, lname_col, fname_col)`
- Processes in batches of 10,000 rows to manage memory.
- Last name set to `""` (empty) since the dataset contains first names only.
- Adds 13 `eth_*` probability columns and `ethnicolr_race` (argmax category).

### Step 2 — langdetect (all names)

- Each name is repeated 5 times before passing to `detect_langs()`.
- `DetectorFactory.seed = 0` ensures reproducible output.
- `LangDetectException` is caught silently; failures produce `None` values.
- Adds `langdetect_lang` (ISO 639-1 code) and `langdetect_prob` (confidence).

### Step 3 — nationalize.io (stratified sample)

- **Sample selection:** 250 names drawn randomly (seed 42) from each of four
  `top_country_prob` bins: [0–0.3], [0.3–0.5], [0.5–0.7], [0.7–1.0]. Equal
  representation across confidence levels avoids bias toward high-confidence
  entries.
- **Deduplication:** Only unique names are queried to avoid wasting API calls.
- **Batch size:** 50 names per HTTP request; 0.5 s delay between batches.
- **Writes:** `agreement_score` and `n_models_used = 1` for sampled rows;
  non-sampled rows remain NaN / 0.
- **Run-time outputs:** A timestamped `.log` file and a structured `.md` report
  are written to `logs/` on every run, capturing configuration, per-step
  statistics, and methodology notes.

---

## 8. File Structure

```
HonsProject/
├── data/
│   ├── names_base.parquet   # Primary dataset (727,545 names, ~16 MB)
│   ├── names.db             # SQLite copy v1 (~1.4 GB)
│   └── names2.db            # SQLite copy v2 (~1.4 GB)
├── logs/                    # Created on first enrich_names.py run
│   ├── enrich_<ts>.log
│   └── enrich_<ts>_report.md
├── notebooks/
│   ├── 01_explore_dataset.ipynb
│   └── 02_explore_names2.ipynb
├── scripts/
│   ├── run-dataset.py
│   ├── database_v1.py
│   ├── database_v2.py
│   ├── parquet_to_sqlite.py
│   ├── playground.py
│   ├── playground_name2nat.py
│   └── enrich_names.py
└── PROJECT_NOTES.md         # This file
```

---

## 9. Key Decisions Log

| Decision | Rationale |
|----------|-----------|
| Parquet as primary format, SQLite as secondary | Parquet is compact, typed, and fast to read with pandas. SQLite is included for ad-hoc SQL querying. |
| `strong_top_country` threshold of 0.7 | Chosen in `database_v2.py` as a round-number boundary separating names with a clear dominant origin from ambiguous ones. 46.7 % of names exceed it. |
| Use `pred_wiki_name` with empty last name | The dataset contains only first names. The wiki model was chosen over the Florida voter-registration model because it has broader international coverage. |
| Repeat name 5× for langdetect | Below ~20 characters, `langdetect` frequently raises `LangDetectException` or returns low-confidence results. Repetition inflates the character sequence artificially without changing its n-gram composition. |
| Stratified sample for nationalize.io | Equal sampling from four probability bins ensures the agreement score is computed over the full range of dataset confidence, not just easy high-confidence cases. |
| Agreement score = nationalize.io probability for `top_country` | A continuous score (rather than binary match/no-match) preserves the strength of agreement. A name where nationalize.io gives 0.4 probability to the same country is more informative than a simple 0/1. |
| name2nat rejected | Confirmed broken on Python 3.11 and 3.13 due to PyTorch GRU serialisation format change. Would require a Python 3.6 virtual environment to use, making it impractical. |
