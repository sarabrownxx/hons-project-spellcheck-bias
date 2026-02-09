import ethnicolr
import pandas as pd
from langdetect import DetectorFactory, detect_langs, detect
from name2nat import Name2nat

# df = pd.DataFrame({"first": ["Adebayo", "Fatima", "José", "Li", "Réka", "Anandi", "Francesca"]})

# df['last'] = ''  # empty last name as last names required

my_nanat = Name2nat()
names = ["Donald Trump", # American
         "Moon Jae-in", # Korean
         "Shinzo Abe", # Japanese
         "Xi Jinping", # Chinese
         "Joko Widodo", # Indonesian
         "Angela Merkel", # German
         "Emmanuel Macron", # French
         "Kyubyong Park", # Korean\
         "Yamamoto Yu", # Japanese
         "Jing Xu"] # Chinese
result = my_nanat(names, top_n=3)
print(result)