import ethnicolr
import pandas as pd
from langdetect import DetectorFactory, detect_langs, detect
from name2nat import Name2nat



df = pd.DataFrame({"first": ["Adebayo", "Fatima", "José", "Li", "Réka", "Anandi", "Francesca"]})

df['last'] = ''  # empty last name as last names required

# Predict ethnicity from ethnicolr wiki model
df_pred = ethnicolr.pred_wiki_name(df, lname_col='last', fname_col='first')

print(df_pred)
# # print(df)
# DetectorFactory.seed = 0
# for name in df['first']:
#     name_string = ''
#     for i in range(5):
#         name_string += name + ' '
#     lang = detect_langs(name_string)
#     print(f'Name: {name}, Language probability: {lang}')