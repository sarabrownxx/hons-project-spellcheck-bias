Honours project - Exploring linguistic bias against names in spellcheck tools

The project is about whether spell checking tools are biased against names from certain cultural or geographic origins, so names like Adebayo or Réka or Fatima getting flagged as misspelled more than names like Sarah or John, or potentially a more meaningful distinction around the origin language of the tool.

The first part is building and exploring the dataset. The data comes from https://github.com/philipperemy/name-dataset? which gives a probability distribution of countries for any first name. database_v2.py pulls all those names into a parquet file, then parquet_to_sqlite.py converts it to a sqlite database (names2.db) which is easier to query. The db files are not on github because they are too large, run those two scripts in that order to regenerate them.

The dataset has about 727k names. each name has its top country, the probability score for that, and a flag called strong_top_country which means the model is more than 70% confident. There are also columns for agreement_score and n_models_used.

The notebooks are the analysis. 01 is early exploration, 02 is the main one - it looks at country distributions, prediction confidence, frequency vs confidence patterns, that kind of thing. The goal is to understand the dataset well before running any spellcheck tests.

The spellcheck testing framework .

playground.py and playground_name2nat.py are just me testing out ethnicolr and name2nat which are libraries that predict name origin, exploring what tools exist

to get back up and running from scratch
1. run scripts/database_v2.py
2. run scripts/parquet_to_sqlite.py
3. open notebooks/02_explore_names2.ipynb

dependencies are pandas, numpy, sqlalchemy, names_dataset, ethnicolr, name2nat
