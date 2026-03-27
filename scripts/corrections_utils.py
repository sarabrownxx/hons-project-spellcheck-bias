import pandas as pd


def build_name_lookup(df: pd.DataFrame) -> dict:
    detail_cols = ["name", "top_country", "name_script",
                   "ethnicolr_race", "langdetect_lang"]
    use_cols = [c for c in detail_cols if c in df.columns]

    sub = df[list({"name", "name_latin"} | set(use_cols))].dropna(subset=["name"]).copy()
    name_arr  = sub["name"].values
    nl_arr    = sub["name_latin"].values
    row_dicts = sub[use_cols].to_dict("records")

    lookup = {}
    for i, (n, nl) in enumerate(zip(name_arr, nl_arr)):
        if nl and nl != n:
            key = nl.lower()
            if key not in lookup:
                lookup[key] = {**row_dicts[i], "matched_via_latin": True}
    for i, n in enumerate(name_arr):
        lookup[n.lower()] = {**row_dicts[i], "matched_via_latin": False}

    return lookup
