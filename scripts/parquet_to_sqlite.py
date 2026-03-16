import pandas as pd
import json
from sqlalchemy import create_engine

PARQUET_DATA_PATH = "data/names_results_base_cleaned.parquet"
# Remember to change filepath for new database
SQLITE_PATH = "data/names_results2.db"

df = pd.read_parquet(PARQUET_DATA_PATH)

# Convert dict/array columns to JSON strings for SQLite
df["full_countries_distribution"] = df["full_countries_distribution"].apply(json.dumps)
df["eth_distribution"] = df["eth_distribution"].apply(json.dumps)
df["top_country_langs"] = df["top_country_langs"].apply(lambda x: json.dumps(list(x)) if x is not None else None)

engine = create_engine(f"sqlite:///{SQLITE_PATH}")

df.to_sql(
    "names",
    engine,
    if_exists="replace",
    index=False
)

print("Created SQLite database at " + SQLITE_PATH)
