"""
Code to merge all data files. 

1. Remove marks with word count >= 5 
2. Add Wordnet Indicator 
3. Add translation: where translation does not change, add Not Applicable 
"""

import pandas as pd
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
nltk.download("wordnet")
nltk.download('omw-1.4')

# Input 
preprocessed_path = '../data/df_preprocessed.pkl'

# Output 
wn_ind_path = '../data/df_wordnet.pkl'

## Read Data 
df_preprocessed = pd.read_pickle(preprocessed_path)

## Remove Marks
def remove_marks(df_preprocessed):
    ## Analyze word count of mark_processed 
    df_preprocessed['mark_len'] = df_preprocessed['mark_processed'].apply(lambda x: len(x.split()))
    # Remove marks with mark_len = 0 
    df_preprocessed = df_preprocessed[df_preprocessed['mark_len']!=0]
    # Remove marks with mark_len >= 5 
    df_preprocessed = df_preprocessed[df_preprocessed['mark_len']<5]
    print("Shape after filtering based on mark_len[1-4]:",df_preprocessed.shape)
    
## Check text in wordnet 
def any_word_in_wn(text):
    wn_ind = 0
    text = text.lower()
    text_list = text.split()
    for txt in text_list:
        if wn.synsets(txt)!=[]:
            wn_ind = 1
            break
    return wn_ind

# Add Wordnet Indicator 
def add_wn_indicator(df_preprocessed):
    df_preprocessed['wn_ind'] = df_preprocessed['mark_processed'].apply(lambda x: any_word_in_wn(x))
    # Analyze word_net indicator
    print("WordNet indicator present in:",np.round(100*df_preprocessed['wn_ind'].sum()/df_preprocessed.shape[0],2),"%")
    # save modelling data
    df_preprocessed.to_pickle(wn_ind_path)


# Remove marks and add wordnet indicator
if __name__ == "__main__":
    remove_marks(df_preprocessed)
    add_wn_indicator(df_preprocessed)