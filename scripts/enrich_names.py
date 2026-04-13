"""
scripts/enrich_names.py

Enriches data/names_base.parquet with three additional signals:

Full dataset (all names):
  ethnicolr_race            – top predicted ethnicity category
  ethnicolr_prob            – probability assigned to ethnicolr_race
  eth_distribution          – dict of all ethnicity probabilities (replaces individual eth_* columns)
  top_country_region        – ethnicolr-aligned region for top_country (e.g. British, EastAsian)
  langdetect_lang           – top ISO 639-1 language code detected from name characters
  langdetect_lang_name      – full language name for langdetect_lang
  langdetect_prob           – confidence of that detection

Sample only (SAMPLE_SIZE names, stratified by top_country_prob; rest are NaN):
  agreement_score           – probability that nationalize.io assigns to top_country
  n_models_used             – 1 for sampled names, 0 otherwise

Usage:
  python scripts/enrich_names.py

  To use a nationalize.io API key (raises daily limit from 100 → 1000):
    set NATIONALIZE_KEY=your_key_here
    python scripts/enrich_names.py
"""

import os
import time
import json
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()  # loads .env locally; no-op on GitHub Actions where secrets are injected directly

import numpy as np
import pandas as pd
import pycountry
from countryinfo import CountryInfo
import requests
import ethnicolr
from langdetect import DetectorFactory, detect_langs, LangDetectException

# Config
INPUT_PATH        = Path("data/names_base.parquet")
PARQUET_PATH      = Path("data/names_results_base.parquet")
SAMPLE_SIZE       = 1000          # Names queried against nationalize.io
ETHNICOLR_BATCH   = 10_000        # Rows per ethnicolr batch (memory vs speed)
LANGDETECT_REPS   = 5             # Repeat name N times for more stable detection
NATIONALIZE_BATCH = 50            # Names per nationalize.io HTTP request
NATIONALIZE_DELAY = 0.5           # Seconds between API batches (respect rate limit)
NATIONALIZE_KEY   = os.getenv("NATIONALIZE_KEY")  # Optional free key

DetectorFactory.seed = 0  # Reproducible langdetect results

# ethnicolr column mapping
ETH_COL_MAP = {
    "Asian,GreaterEastAsian,EastAsian":       "eth_EastAsian",
    "Asian,GreaterEastAsian,Japanese":        "eth_Japanese",
    "Asian,IndianSubContinent":               "eth_IndianSubContinent",
    "GreaterAfrican,Africans":                "eth_African",
    "GreaterAfrican,Muslim":                  "eth_Muslim",
    "GreaterEuropean,British":                "eth_British",
    "GreaterEuropean,EastEuropean":           "eth_EastEuropean",
    "GreaterEuropean,Jewish":                 "eth_Jewish",
    "GreaterEuropean,WestEuropean,French":    "eth_French",
    "GreaterEuropean,WestEuropean,Germanic":  "eth_Germanic",
    "GreaterEuropean,WestEuropean,Hispanic":  "eth_Hispanic",
    "GreaterEuropean,WestEuropean,Italian":   "eth_Italian",
    "GreaterEuropean,WestEuropean,Nordic":    "eth_Nordic",
}

# Short label for each full ethnicolr race string (used for ethnicolr_prob)
RACE_SHORT = {full: col.replace("eth_", "") for full, col in ETH_COL_MAP.items()}


# Country name → ISO alpha-2
def build_country_map():
    """Build a lookup dict from the country name variants pycountry knows about."""
    mapping = {}
    for c in pycountry.countries:
        mapping[c.name] = c.alpha_2
        if hasattr(c, "common_name"):
            mapping[c.common_name] = c.alpha_2
        if hasattr(c, "official_name"):
            mapping[c.official_name] = c.alpha_2
    return mapping


def to_iso2(country_name, country_map):
    """Return ISO alpha-2 for a country name, or None if unresolvable."""
    if country_name in country_map:
        return country_map[country_name]
    try:
        results = pycountry.countries.search_fuzzy(country_name)
        return results[0].alpha_2 if results else None
    except LookupError:
        return None


# Helpers
def _lang_name(code):
    """Convert an ISO 639-1 language code (e.g. 'fr', 'zh-cn') to a full name."""
    if not code or pd.isna(code):
        return None
    lang = pycountry.languages.get(alpha_2=str(code).split("-")[0])
    return lang.name if lang else code


def _country_langs(country_name, country_map):
    """Return list of ISO 639-1 language codes for a country, via countryinfo."""
    iso2 = to_iso2(country_name, country_map)
    if not iso2:
        return []
    try:
        return CountryInfo(iso2).languages() or []
    except Exception:
        return []


def _consolidate_ethnicolr(df):
    """
    After run_ethnicolr, adds two derived columns then drops the 13 individual
    eth_* probability columns (which are now captured in eth_distribution).

    Columns added (ethnicolr_race is kept as-is):
      ethnicolr_prob  : probability of the top-predicted race (from ethnicolr_race)
      eth_distribution: dict {short_race: probability} for all 13 categories
    """
    eth_col_names = list(ETH_COL_MAP.values())
    short_names   = [c.replace("eth_", "") for c in eth_col_names]

    # ethnicolr_prob: pick the probability column matching each row's ethnicolr_race
    df["ethnicolr_prob"] = np.nan
    for full_race, col in ETH_COL_MAP.items():
        mask = df["ethnicolr_race"] == full_race
        if mask.any():
            df.loc[mask, "ethnicolr_prob"] = df.loc[mask, col]

    # eth_distribution: consolidate all 13 eth_* probabilities into one dict per row
    eth_arr = df[eth_col_names].to_numpy(dtype=float)
    df["eth_distribution"] = [
        {k: round(float(v), 4) for k, v in zip(short_names, row) if not np.isnan(v)}
        for row in eth_arr
    ]

    df.drop(columns=eth_col_names, inplace=True)


# Step 1: ethnicolr
def _run_ethnicolr_on_column(df, name_col):
    """
    Run ethnicolr on a name column and return a DataFrame of predictions
    aligned to all rows, with the original index preserved.
    """
    idx    = df.index
    n      = len(idx)
    n_batches = (n + ETHNICOLR_BATCH - 1) // ETHNICOLR_BATCH
    results = []

    for i in range(n_batches):
        batch_idx = idx[i * ETHNICOLR_BATCH : (i + 1) * ETHNICOLR_BATCH]
        chunk = pd.DataFrame({
            "first": df.loc[batch_idx, name_col].values,
            "last":  "",
        })
        try:
            pred = ethnicolr.pred_wiki_name(chunk, lname_col="last", fname_col="first")
            pred.index = batch_idx
            results.append(pred)
        except Exception as exc:
            print(f"  WARNING: ethnicolr batch {i + 1} failed – {exc}", flush=True)

    return pd.concat(results) if results else pd.DataFrame()


def run_ethnicolr(df):
    """
    Runs ethnicolr pred_wiki_name in batches of ETHNICOLR_BATCH.
    Adds eth_* probability columns and ethnicolr_race to df.

    Pass 1: uses the original `name` column (all rows).
    Pass 2: for rows where ethnicolr returned NaN (non-Latin names that were
            skipped after ASCII normalisation), re-runs using `name_latin`
            if that column exists in df.  This fills in predictions for the
            ~18% of names in Arabic, CJK, Cyrillic, etc. scripts.
    """
    n = len(df)

    # Pre-allocate output columns with NaN
    for col in ETH_COL_MAP.values():
        df[col] = np.nan
    df["ethnicolr_race"] = pd.NA

    # Pass 1: original names
    print(f"  Pass 1: original names ({n:,} rows in {(n + ETHNICOLR_BATCH - 1) // ETHNICOLR_BATCH} batches)…", flush=True)
    pred1 = _run_ethnicolr_on_column(df, "name")
    if not pred1.empty:
        for raw_col, clean_col in ETH_COL_MAP.items():
            if raw_col in pred1.columns:
                df.loc[pred1.index, clean_col] = pred1[raw_col].values
        if "race" in pred1.columns:
            df.loc[pred1.index, "ethnicolr_race"] = pred1["race"].values

    # Pass 2: name_latin for non-Latin names
    if "name_latin" not in df.columns:
        print("  name_latin column not found — skipping second pass. "
              "Run preprocess_names.py first for full coverage.", flush=True)
        return df

    non_latin_mask = df["ethnicolr_race"].isna()
    n_second = int(non_latin_mask.sum())
    if n_second == 0:
        print("  No NaN ethnicolr rows — second pass not needed.", flush=True)
        return df

    print(f"  Pass 2: {n_second:,} non-Latin names → using name_latin…", flush=True)
    non_latin_idx = df.index[non_latin_mask]
    n2 = len(non_latin_idx)
    n2_batches = (n2 + ETHNICOLR_BATCH - 1) // ETHNICOLR_BATCH
    print(f"    {n2:,} names in {n2_batches} batches…", flush=True)

    for i in range(n2_batches):
        batch_idx = non_latin_idx[i * ETHNICOLR_BATCH : (i + 1) * ETHNICOLR_BATCH]
        chunk = pd.DataFrame({
            "first": df.loc[batch_idx, "name_latin"].values,
            "last":  "",
        })
        try:
            pred = ethnicolr.pred_wiki_name(chunk, lname_col="last", fname_col="first")
            for raw_col, clean_col in ETH_COL_MAP.items():
                if raw_col in pred.columns:
                    df.loc[batch_idx, clean_col] = pred[raw_col].values
            if "race" in pred.columns:
                df.loc[batch_idx, "ethnicolr_race"] = pred["race"].values
        except Exception as exc:
            print(f"  WARNING: ethnicolr pass-2 batch {i + 1} failed – {exc}", flush=True)

    still_nan = int(df["ethnicolr_race"].isna().sum())
    print(f"  After both passes: {still_nan:,} rows still NaN "
          f"(names that are empty after anyascii transliteration)", flush=True)

    return df


# Step 2: langdetect
def _detect_one(name):
    """Return (lang_code, probability) for a single name, or (None, None) on failure."""
    try:
        text = (name + " ") * LANGDETECT_REPS
        langs = detect_langs(text)
        if langs:
            return langs[0].lang, round(langs[0].prob, 4)
    except LangDetectException:
        pass
    return None, None


def run_langdetect(df):
    """
    Applies langdetect to every name.
    Adds langdetect_lang and langdetect_prob columns.
    Note: noisy for short/ambiguous names — treat as a supporting signal only.
    """
    print(f"  Detecting language for {len(df):,} names (this may take ~15 min)…", flush=True)
    results = df["name"].apply(lambda n: pd.Series(_detect_one(n),
                                                   index=["langdetect_lang",
                                                          "langdetect_prob"]))
    df["langdetect_lang"] = results["langdetect_lang"]
    df["langdetect_prob"] = pd.to_numeric(results["langdetect_prob"], errors="coerce")
    df["langdetect_lang_name"] = df["langdetect_lang"].apply(_lang_name)
    return df


# Step 3: nationalize.io sample
def _fetch_nationalize(names, api_key=None):
    """
    POST a batch of names to nationalize.io.
    Returns dict: name → list[{country_id, probability}].
    """
    params = [("name[]", n) for n in names]
    if api_key:
        params.append(("apikey", api_key))
    try:
        resp = requests.get("https://api.nationalize.io", params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):   # single-name response
            data = [data]
        return {item["name"]: item.get("country", []) for item in data}
    except Exception as exc:
        print(f"  WARNING: nationalize.io request failed – {exc}")
        return {}


def run_nationalize_sample(df, country_map):
    """
    Selects a stratified sample of SAMPLE_SIZE names (quartiles of top_country_prob),
    queries nationalize.io, and writes agreement_score / n_models_used.
    """
    # Stratified sample across confidence levels
    bins   = [0.0, 0.3, 0.5, 0.7, 1.01]
    labels = ["0–0.3", "0.3–0.5", "0.5–0.7", "0.7–1.0"]
    df["_bin"] = pd.cut(df["top_country_prob"], bins=bins, labels=labels,
                        include_lowest=True)
    per_bin = SAMPLE_SIZE // len(labels)

    sample_idx = (
        df.groupby("_bin", observed=True)
          .apply(lambda g: g.sample(min(len(g), per_bin), random_state=42))
          .index.get_level_values(1)
    )
    df.drop(columns=["_bin"], inplace=True)

    print(f"  Sampled {len(sample_idx):,} names for nationalize.io", flush=True)

    # Pre-resolve top_country → ISO alpha-2 for sampled rows only
    iso2_lookup = {
        idx: to_iso2(df.at[idx, "top_country"], country_map)
        for idx in sample_idx
    }

    # Query API in batches (unique names only to avoid duplicate requests)
    unique_names = list(df.loc[sample_idx, "name"].unique())
    n_batches    = (len(unique_names) + NATIONALIZE_BATCH - 1) // NATIONALIZE_BATCH
    nat_results  = {}

    for i in range(n_batches):
        batch = unique_names[i * NATIONALIZE_BATCH:(i + 1) * NATIONALIZE_BATCH]
        nat_results.update(_fetch_nationalize(batch, api_key=NATIONALIZE_KEY))
        if i < n_batches - 1:
            time.sleep(NATIONALIZE_DELAY)
        if (i + 1) % 10 == 0 or (i + 1) == n_batches:
            print(f"    batch {i + 1}/{n_batches}", flush=True)

    # Write scores back
    for idx in sample_idx:
        name      = df.at[idx, "name"]
        iso2      = iso2_lookup[idx]
        preds     = nat_results.get(name, [])
        score     = next(
            (p["probability"] for p in preds if p["country_id"] == iso2),
            0.0,
        )
        df.at[idx, "agreement_score"] = round(score, 4)
        df.at[idx, "n_models_used"]  += 1

    matched    = sum(1 for idx in sample_idx if df.at[idx, "agreement_score"] > 0)
    mean_score = df.loc[sample_idx, "agreement_score"].mean()
    print(f"  Non-zero agreement: {matched}/{len(sample_idx)} names  "
          f"(mean score: {mean_score:.3f})", flush=True)

    return df


# Main
def main():
    print(f"Loading {INPUT_PATH}…", flush=True)
    df = pd.read_parquet(INPUT_PATH)
    print(f"  {len(df):,} names loaded\n")

    country_map = build_country_map()

    def checkpoint(label):
        """Save current state of df to parquet immediately after a step completes.
        If a later step crashes, this checkpoint is already on disk and will be
        captured by the GitHub Actions artifact upload (if: always())."""
        print(f"  Checkpoint: saving after {label}…", flush=True)
        df.to_parquet(PARQUET_PATH, index=False)
        print(f"  Checkpoint saved to {PARQUET_PATH}", flush=True)

    # 1. ethnicolr
    print("[1/3] ethnicolr — running on all names…")
    t0 = time.time()
    df = run_ethnicolr(df)
    print(f"  Done in {time.time() - t0:.0f}s")
    _consolidate_ethnicolr(df)
    df["n_models_used"] += (~df["ethnicolr_race"].isna()).astype(int)
    checkpoint("ethnicolr")

    # 2. langdetect
    print("\n[2/3] langdetect — running on all names…")
    t0 = time.time()
    df = run_langdetect(df)
    print(f"  Done in {time.time() - t0:.0f}s")
    df["n_models_used"] += (~df["langdetect_lang"].isna()).astype(int)
    checkpoint("langdetect")

    # 3. nationalize.io
    print("\n[3/3] nationalize.io — querying sample…")
    if NATIONALIZE_KEY:
        print(f"  Using API key (1 000 req/day limit)")
    else:
        print(f"  No API key set — limit is 100 req/day.  "
              f"Set NATIONALIZE_KEY env var for 1 000/day.")
    t0 = time.time()
    df = run_nationalize_sample(df, country_map)
    print(f"  Done in {time.time() - t0:.0f}s\n")

    # Country language comparison
    print("\nAdding top_country_langs and country_lang_comp…", flush=True)
    unique_countries = df["top_country"].dropna().unique()
    langs_lookup = {c: _country_langs(c, country_map) for c in unique_countries}
    df["top_country_langs"] = df["top_country"].map(langs_lookup)
    df["country_lang_comp"] = df.apply(
        lambda row: bool(row["langdetect_lang"]) and
                    row["langdetect_lang"] in (row["top_country_langs"] or []),
        axis=1,
    )

    # Column ordering
    desired_order = [
        "name",
        "full_countries_distribution",
        "top_country", "top_country_langs", "top_country_prob", "strong_top_country",
        "agreement_score", "n_models_used",
        "name_script", "name_latin",
        "ethnicolr_race", "ethnicolr_prob", "eth_distribution",
        "langdetect_lang", "langdetect_lang_name", "langdetect_prob",
        "country_lang_comp",
    ]
    ordered = [c for c in desired_order if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered]
    df = df[ordered + remaining]

    # Final save
    print(f"Saving to {PARQUET_PATH}…", flush=True)
    df.to_parquet(PARQUET_PATH, index=False)
    print("Saved.\n")

    # Summary
    print("── ethnicolr top category distribution (sample of 10) ──")
    print(df["ethnicolr_race"].value_counts().head(10).to_string())

    print("\n── langdetect top language distribution (sample of 10) ──")
    print(df["langdetect_lang"].value_counts().head(10).to_string())

    print(f"\n── country_lang_comp: {df['country_lang_comp'].sum():,} / {df['country_lang_comp'].notna().sum():,} names match ──")

    sampled = df[df["n_models_used"] > 0]
    print(f"\n── agreement_score (n={len(sampled):,}) ──")
    print(sampled["agreement_score"].describe().to_string())


if __name__ == "__main__":
    main()
