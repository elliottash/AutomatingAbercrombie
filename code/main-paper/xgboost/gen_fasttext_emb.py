'''
Code to generate Fasttext embedding (difference of statement and mark_id)
'''

import pickle 
import numpy as np
import pandas as pd
import fasttext as ft


# Input 
modelling_base = '../data/distilbert/df_bert_input.pkl'

# Output 
fasttext_emb_diff = '../data/xgboost/df_ft_diff.pkl'


df_modelling_ft = pd.read_pickle(modelling_base)
df_ft = df_modelling_ft[['serial_no','statement_processed','mark_processed']]


'''
Create fastText embeddings for statement_processed. 
'''
get_ipython().system('git clone https://github.com/facebookresearch/fastText.git')
get_ipython().system('cd fastText')
get_ipython().system('pip install fastText')

import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')  # English
model = fasttext.load_model('cc.en.300.bin')


## Get fasttext embeddings for statement and marks. 
statement_embedding = df_ft['statement_processed'].apply(lambda x: model.get_sentence_vector(x))
mark_id_embedding = df_ft['mark_processed'].apply(lambda x: model.get_sentence_vector(x))

## Get difference of statement and mark embeddings.
embedding_diff = statement_embedding-mark_id_embedding
df_embedding_diff = pd.DataFrame.from_dict(dict(zip(embedding_diff.index, embedding_diff.values))).T
print("Embedding-difference shape:",df_embedding_diff.shape)

## Change column names 
col_list = []
for i in range(300):
    col_list.append('emb_dim_'+str(i))
df_embedding_diff.columns = col_list

print(df_embedding_diff.head())


df_embedding_diff['serial_no'] = df_ft['serial_no']
df_embedding_diff.to_pickle(fasttext_emb_diff)



