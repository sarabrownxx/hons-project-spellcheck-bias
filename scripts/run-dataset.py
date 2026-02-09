from names_dataset import NameDataset, NameWrapper
import pandas as pd

# Takes a bit to load, do this once at the start of a notebook/script.
nd = NameDataset()

res = nd.search("Philippe")
print(res)

print(NameWrapper(res).describe)   # e.g. "Male, France"

# Create empty Pandas DataFrame with specified columns
df = pd.DataFrame(
    columns=[
        "name",
        "full_countries_distribution",
        "top_country",
        "top_country_prob",
        "strong_top_country",  # whether top country prob > 0.7
        # continent or sub-continent level may be helpful to later spot global patterns
        # proximity to spellchecker country of origin
        "agreement_score", # how much the various models agree with top country of origin
        "n_models_used",
        # flag for whether each tool flagged the name as incorrect
        # top suggested corrections from each tool and whether they were found in the dataset so likely to be a known name
    ]
)
