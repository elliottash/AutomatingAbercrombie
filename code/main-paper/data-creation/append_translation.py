'''
Code to translate mark_id 
'''

get_ipython().system('pip install googletrans==3.1.0a0')

import numpy as np
import pandas as pd 
from googletrans import Translator
from time import time 

# Input 
model_data_path = '../data/df_modelling_base_prot4.pkl'

# Output 
out_path = '../data/df_translated.csv'
final_output_path = '../data/df_translated_final.pkl'


df_input = pd.read_pickle(model_data_path)
translator = Translator()

# Function to translate mark text
def translate_mark(txt):
    result = translator.translate(txt)
    if result.src!='en':
        return result.text 
    return "-1" 

n = 100 
list_df = np.array_split(df_input, n)

# Translate in batches 
for i in range(n):
    start = time()
    df_translated = list_df[i][['serial_no','filing_dt']].copy()
    df_translated['mark_translated'] = list_df[i]['mark_processed'].apply(lambda x: translate_mark(x))
    if i==0:
        df_translated.to_csv(out_path, index=False)
    else:
        df_translated.to_csv(out_path, index=False, mode='a', header=False)
    end = time()
    print("Chunk: ",i)
    print("Time taken: ", str((end-start)/60) + " min") 


# Merge with modeling data
df_processed = pd.read_csv(out_path)
df_merged = df_input.merge(df_processed, on = ['serial_no','filing_dt'], how = 'inner')

# Select translations
def select_mark(x):
    if x['mark_translated'] == '-1':
        return x['mark_processed']
    return x['mark_translated']


# Process and dump file.
df_merged['mark_final'] = df_merged.apply(select_mark, axis=1)
print(df_merged.shape)
df_translated_final = df_merged[['serial_no','filing_dt','mark_final']].copy()
df_translated_final.to_pickle(final_output_path, protocol=4)