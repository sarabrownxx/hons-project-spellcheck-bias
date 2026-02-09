import pandas as pd
import numpy as np
from names_dataset import NameDataset

nd = NameDataset()

rows = []

for name in nd.first_names.keys():
    res = nd.search(name)
    fn = res.get("first_name")

    if not fn or "country" not in fn:
        continue

    countries = fn["country"]
    if not countries:
        continue

    top_country, top_prob = max(countries.items(), key=lambda x: x[1])

    rows.append({
        "name": name,
        "full_countries_distribution": countries,
        "top_country": top_country,
        "top_country_prob": top_prob,
        "strong_top_country": top_prob > 0.7,
        "agreement_score": np.nan,
        "n_models_used": 0,
    })


df = pd.DataFrame(rows)
df["strong_top_country"] = df["strong_top_country"].astype(bool)  # Explicitly cast the strong_top_country column to boolean before saving
print("Saving Parquet file...")
df.to_parquet("data/names_base.parquet")
print("Parquet file saved successfully.")