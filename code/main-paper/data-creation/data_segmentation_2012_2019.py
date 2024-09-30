'''
This code picks up case file data from 2012-2019 (inclusive) on the basis of filing_dt 
and dumps it to a given location. 
'''

import pandas as pd 
import numpy as np
import os 


# Input 
case_files = '../../base_data/case_file.csv'

# Output 
case_files_2012_2019 = '../data/case_file_2012_2019.csv'



# Write data from 2012-2019 to location 

df_cf = pd.read_csv(case_files, nrows = 2)
cols_to_write = list(df_cf.columns)
df_iterator = pd.read_csv(case_files, chunksize=10000)

count = 0 
for i, df_chunk in enumerate(df_iterator):
    df_chunk['filing_dt'] = pd.to_datetime(df_chunk['filing_dt'])
    df_chunk_to_write = df_chunk[df_chunk['filing_dt'].apply(lambda x: x.year>=2012 and x.year<=2019)]
    if count==0:
        df_chunk_to_write.to_csv(case_files_2012_2019, index=False, mode='w', header = cols_to_write)
    else:
        df_chunk_to_write.to_csv(case_files_2012_2019, index=False, mode='a', header = None)
    count += 1


df_cf_2012 = pd.read_csv(case_files_2012_2019)
print(df_cf_2012.shape)
print(df_cf_2012['filing_dt'].min())
print(df_cf_2012['filing_dt'].max())

