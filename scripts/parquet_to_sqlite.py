import pandas as pd
import json
from sqlalchemy import create_engine

PARQUET_DATA_PATH = "data/final_results.parquet"
# Remember to change filepath for new database
SQLITE_PATH = "data/full_results1.db"

df = pd.read_parquet(PARQUET_DATA_PATH)

# Convert dict/array columns to JSON strings for SQLite
df["full_countries_distribution"] = df["full_countries_distribution"].apply(json.dumps)
df["eth_distribution"] = df["eth_distribution"].apply(json.dumps)
df["top_country_langs"] = df["top_country_langs"].apply(lambda x: json.dumps(list(x)) if x is not None else None)

match_cols = [
    "hunspell_orig_correction_match", "hunspell_latin_correction_match",
    "symspell_orig_correction_match", "symspell_latin_correction_match",
    "lt_orig_correction_match",       "lt_latin_correction_match",
    "pysc_orig_correction_match",     "pysc_latin_correction_match",
]
for col in match_cols:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: json.dumps(x) if x is not None else None)

engine = create_engine(f"sqlite:///{SQLITE_PATH}")

df.to_sql(
    "names",
    engine,
    if_exists="replace",
    index=False
)

print("Created SQLite database at " + SQLITE_PATH)
