import pandas as pd
import json
from sqlalchemy import create_engine

PARQUET_DATA_PATH = "data/names_base.parquet"
# Remember to change filepath for new database
SQLITE_PATH = "data/names2.db"

df = pd.read_parquet(PARQUET_DATA_PATH)

# Convert the 'full_countries_distribution' column to JSON string for SQLite
df["full_countries_distribution"] = df["full_countries_distribution"].apply(json.dumps)

engine = create_engine(f"sqlite:///{SQLITE_PATH}")

df.to_sql(
    "names",
    engine,
    if_exists="replace",
    index=False
)

print("Created SQLite database at " + SQLITE_PATH)
