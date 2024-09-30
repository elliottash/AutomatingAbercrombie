"""
Code to preprocess statement for BERT Input. 
1. Preprocess statement_text for all statement types.
2. Remove Pseudo Marks, and dump it. 
3. Join all other statement types. 
"""

import pandas as pd 
import numpy as np 
import pickle 
import re 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from collections import Counter

# Input 
statement_path = '../../base_data/statement.csv'
preprocessed_data = '../data/df_preprocessed.pkl'

# Output 
psuedo_mark_path = '../../base_data/psuedo_marks.pkl'
psuedo_mark_unprocessed_path = '../../base_data/psuedo_marks_unprocessed.pkl'
statement_no_pm_2012_path = '../../base_data/statement_no_pm_2012_2019.pkl'
statement_no_pm_2012_unprocessed_path = '../../base_data/statement_no_pm_unprocessed_2012_2019.pkl'

## Read input data 
def read_data(statement_path, preprocessed_data):
    # Get serial no from preprocessed data 
    s_no = pd.read_pickle(preprocessed_data)['serial_no'].tolist()
    # Get corresponding statement text 
    df_statement = pd.read_csv(statement_path)
    df_statement = df_statement[df_statement['serial_no'].isin(s_no)]
    return df_statement

# Preprocess statement
def preprocess_statement(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text 

# Function to separate pseudo marks from statement text 
def create_pm_data(df_statement):
    # Create DF with pseudo marks 
    df_pm = df_statement[df_statement['statement_type_cd'].apply(lambda x: str(x).strip().startswith('PM'))]
    df_pm.reset_index(drop=True, inplace=True)
    df_pm['statement_processed'] = df_pm['statement_processed'].apply(str)
    df_pm['statement_unprocessed'] = df_pm['statement_unprocessed'].apply(str)

    # Join all statements in Pseudo mark - for processed and unprocessed statements 
    df_pm_agg = df_pm[['statement_processed','serial_no']].groupby(['serial_no'], as_index=False).agg({'statement_processed':' '.join})
    df_pm_unprocessed_agg = df_pm[['statement_unprocessed','serial_no']].groupby(['serial_no'], as_index=False).agg({'statement_unprocessed':' '.join})

    print("Shape of Pseudo Marks: ", df_pm_agg.shape)
    df_pm_agg.to_pickle(psuedo_mark_path)
    df_pm_unprocessed_agg.to_pickle(psuedo_mark_unprocessed_path)
    return df_pm_agg, df_pm_unprocessed_agg

# Function to get df with no pseudo marks 
def create_no_pm_data(df_statement):
    # Statements with no pseudo marks 
    df_statement_no_pm = df_statement[df_statement['statement_type_cd'].apply(lambda x: (str(x).strip().startswith('PM')==False))]
    df_statement_no_pm.reset_index(drop=True, inplace=True)
    
    # Join all statements 
    df_statement_no_pm['statement_processed'] = df_statement_no_pm['statement_processed'].apply(str)
    df_statement_no_pm['statement_unprocessed'] = df_statement_no_pm['statement_unprocessed'].apply(str)
    
    df_statement_no_pm_agg = df_statement_no_pm[['statement_processed','serial_no']].groupby(['serial_no'], as_index=False).agg({'statement_processed':' '.join})
    df_statement_no_pm_unprocessed_agg = df_statement_no_pm[['statement_unprocessed','serial_no']].groupby(['serial_no'], as_index=False).agg({'statement_unprocessed':' '.join})

    print("Shape of statements without Pseudo Marks: ", df_statement_no_pm_agg.shape)
    df_statement_no_pm_agg.to_pickle(statement_no_pm_2012_path)
    df_statement_no_pm_unprocessed_agg.to_pickle(statement_no_pm_2012_unprocessed_path)

    return df_statement_no_pm_agg, df_statement_no_pm_unprocessed_agg


if __name__ == "__main__": 
    df_statement = read_data(statement_path, preprocessed_data)
    df_statement['statement_unprocessed'] = df_statement['statement_text'].apply(str).copy()
    df_statement['statement_processed'] = df_statement['statement_text'].apply(lambda x: preprocess_statement(x))
    df_pm_agg, df_pm_unprocessed_agg = create_pm_data(df_statement)
    df_statement_no_pm_agg, df_statement_no_pm_unprocessed_agg = create_no_pm_data(df_statement)






