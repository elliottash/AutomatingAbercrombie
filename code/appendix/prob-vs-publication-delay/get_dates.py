"""
Code to get different dates for mark - 
Index column = serial_no, mark_id_char
"""

import pandas as pd 
import numpy as np 


## Input data path
case_file_2012_2019 = '../../data/case_file_2012_2019.csv'
## Output data path
mark_dates_path = '../../base_data/df_marks_date.csv'

## Read data 
def read_data(input_path=case_file_2012_2019):
    df = pd.read_csv(input_path)
    return df 

## Get all date columns 
def get_date_cols(df):
    all_cols = list(df.columns)
    date_cols = [x for x in all_cols if 'dt' in x]
    return date_cols

## Subset data - take serial_no, mark_id_char and all date columns 
def get_subset(df, date_cols):
    idx_cols = ['serial_no', 'mark_id_char']
    df_subset = df[idx_cols + date_cols].copy()
    ## Check unique rows 
    print("DF Subset Shape: ", df_subset.shape)
    print("DF Subset unique Shape: ", df_subset.drop_duplicates().shape)
    print("Unique on mark level: ", df_subset[['serial_no','mark_id_char']].drop_duplicates().shape)
    return df_subset

if __name__ == "__main__":
    df = read_data()
    date_cols = get_date_cols(df)
    df_subset = get_subset(df, date_cols)
    df_subset.to_csv(mark_dates_path, index=False)