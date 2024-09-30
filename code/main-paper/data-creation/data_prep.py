import pandas as pd
import numpy as np
import os
from functools import reduce
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')
from collections import Counter
import pickle



'''
Code to prepare dataset (y variable) for distinctiveness indicator. 
1. Data Input - case files from 2012
2. Create y-variable (1): Inherently distinctive - 
    a. publication_dt available (publication_ind = True) and acq_dist_ind = False
    b. publication_dt not available (publication_ind = False), supp_reg_in = False, descriptive_sum = 0, generic_sum = 0 
3. Create y-variable (0): Not distinctive - 
    a. publication_ind = True, and acq_dist_ind = True 
    b. publication_ind = False, and (descriptive_sum != 0 or  generic_sum != 0) 
    c. supp_reg_in = True 
4. Get statistics of each condition for y-variable 
5. Analyze marks on the basis of - 1 word, 2 words, 3 words, 4+ 
6. Drop marks with images. 
7. Final data - serial_no, registration_no, publication_dt, filing_dt, statement_text, mark_id_char,
                acq_dist_in, acq_dist_part_in, serv_mark_in, trade_mark_in, supp_reg_ind, distinctiveness_ind 
6. Add NICE Classes and statement_text 
7. Data Pre-processing: No preprocessing for statement text 
'''


# Input Data 
statement = '../../base_data/statement.csv'
case_file_2012_2019 = '../../reg_indicator/data/case_file_2012_2019.csv'
design_file = '../../base_data/design_search.csv'
nice_classes = '../../base_data/intl_class.csv'
office_actions = '../data/event_GNFR_GNRT_2003-2019_desc_generic_sum.csv'

# Output 
preprocessed_data = '../data/df_preprocessed.pkl'
no_preprocessed_data_path = '../data/df_nopreprocessed.pkl'

# Read all datasets 
def read_data(case_file_2012_2019, statement, design_file, nice_classes, office_actions):
    # Read case files 
    print("Reading raw data")
    cf_usecols = ['serial_no','filing_dt','registration_dt','publication_dt','registration_no','acq_dist_in', 'acq_dist_part_in',
                'serv_mark_in', 'trade_mark_in', 'mark_id_char', 'draw_color_file_in', 'draw_3d_file_in', 'supp_reg_in', 'amend_supp_reg_in']
    df_cf_2012 = pd.read_csv(case_file_2012_2019, usecols=cf_usecols)
    print("Case Files: ", df_cf_2012.shape)
    print("Null in Case Files: ", df_cf_2012.isnull().sum())
    
    # Read design data 
    df_design = pd.read_csv(design_file)

    # Read statement data 
    df_statement = pd.read_csv(statement)

    # Read NICE Classes 
    df_classes = pd.read_csv(nice_classes)

    # Read Office Actions 
    df_office_actions = pd.read_csv(office_actions)
    df_office_actions['serial_no'] = df_office_actions['serial_no'].apply(str).apply(lambda x: x.strip())
    df_office_actions['serial_no'] = pd.to_numeric(df_office_actions['serial_no'])
    df_office_actions.dropna(inplace=True)
    df_office_actions.reset_index(drop=True, inplace=True)
    df_office_actions['serial_no'] = df_office_actions['serial_no'].astype(np.int64)

    return df_cf_2012, df_design, df_statement, df_classes, df_office_actions
    

# Drop entries with mark_id_char = null 
def drop_null_marks(df_cf_2012):
    print("Dropping marks with mark_id_char = null")
    print("Shape before dropping null marks: ", df_cf_2012.shape)
    df_cf_2012.dropna(subset = ['mark_id_char'], inplace=True)
    df_cf_2012.reset_index(drop=True, inplace=True)
    print(df_cf_2012.shape)
    return df_cf_2012


# Drop marks with design - images 
def drop_images(df_design, df_cf_2012):
    print("Drop marks with images")
    print("Shape before dropping Images: ", df_cf_2012.shape)
    drop_serial_no = list(set(df_design['serial_no']).intersection(set(df_cf_2012['serial_no'])))
    df_cf_2012 = df_cf_2012[~df_cf_2012['serial_no'].isin(drop_serial_no)]
    df_cf_2012.reset_index(drop=True,inplace=True)
    print("Shape after dropping Images: ", df_cf_2012.shape)
    return df_cf_2012

# Drop duplicates 
def drop_duplicates(df_cf_2012):
    print("Drop duplicates")
    print("Shape before dropping duplicates: ", df_cf_2012.shape)
    df_cf_2012.sort_values(by=['filing_dt'], ignore_index=True, inplace=True)
    df_cf_2012.drop_duplicates(subset=['mark_id_char'], inplace=True, keep='last')
    df_cf_2012.reset_index(drop=True, inplace=True)
    print("Shape after dropping duplicates: ", df_cf_2012.shape)
    print("Null values: ", df_cf_2012.isnull().sum())
    return df_cf_2012

## Creating distinctiveness indicator 
## Create y-variable (1): Inherently distinctive - 
##     a. publication_dt available (publication_ind = True) and acq_dist_ind = False
##     b. publication_dt not available (publication_ind = False), supp_reg_in = False, descriptive_sum = 0, generic_sum = 0 
## Create y-variable (0): Not distinctive - 
##     a. publication_ind = True, and acq_dist_ind = True 
##     b. publication_ind = False, and (descriptive_sum != 0 or  generic_sum != 0) 
##     c. supp_reg_in = True 
def create_y_variable(df_cf_2012, df_office_actions):
    print("Creating distinctiveness indicator")
    df_cf_2012['distinct_ind'] = -1 
    df_cf_2012['pub_ind'] = (df_cf_2012['publication_dt'].isnull() == False) * 1 
    # Logic (1)a 
    df_cf_2012['distinct_ind'][(df_cf_2012['pub_ind']==1) & (df_cf_2012['acq_dist_in']==0)] = 1 
    # Logic (1)b
    # Merge Case Files and Office Actions
    df_cf_2012 = df_cf_2012.merge(df_office_actions, on = ['serial_no'], how = 'left')
    print("Incidence Rate after logic (1)a: ", (df_cf_2012['distinct_ind']==1).sum()/df_cf_2012.shape[0])
    # Serial numbers receiving no Office Action will have NaN in generic_sum and descriptive_sum 
    # Check how many serial nos got an office action (sanity check)
    oa_count = df_cf_2012.shape[0] - df_cf_2012['generic_sum'].isnull().sum()
    oa_perc = np.round(100*oa_count/df_cf_2012.shape[0],2) 
    print("Percentage of applications receiving Office Actions: ", oa_perc)
    df_cf_2012['distinct_ind'][ (df_cf_2012['pub_ind']==0) & (df_cf_2012['generic_sum']==0) & (df_cf_2012['descriptive_sum']==0) & (df_cf_2012['supp_reg_in']==0) ] = 1 
    print("Cum. Incidence Rate after logic (1)b: ", (df_cf_2012['distinct_ind']==1).sum()/df_cf_2012.shape[0] )
    # Logic (0)a 
    df_cf_2012['distinct_ind'][ (df_cf_2012['pub_ind']==1) & (df_cf_2012['acq_dist_in']==1) ] = 0 
    # Logic (0)b 
    df_cf_2012['distinct_ind'][ (df_cf_2012['pub_ind']==0) & (df_cf_2012['descriptive_sum']!=0) ] = 0 
    df_cf_2012['distinct_ind'][ (df_cf_2012['pub_ind']==0) & (df_cf_2012['generic_sum']!=0) ] = 0 
    # Logic (0)c 
    df_cf_2012['distinct_ind'][(df_cf_2012['supp_reg_in']==1)] = 0
    print("Incidence rates before converting -1 to 0: ")
    print(df_cf_2012['distinct_ind'].value_counts(normalize=True) * 100) 
    # Logic - convert remaining -1 to 0s 
    df_cf_2012['distinct_ind'][(df_cf_2012['distinct_ind']==-1)] = 0 
    print("Incidence rates: ")
    print(df_cf_2012['distinct_ind'].value_counts(normalize=True) * 100) 
    return df_cf_2012


# Get NICE Classes 
def get_nice_classes(df_classes, df_cf_2012):
    print("Getting required NICE Classes")
    # Selecting records with only 1 entry for NICE Classes. 
    df_classes = df_classes[df_classes['serial_no'].isin(list(df_cf_2012['serial_no']))]
    df_classes_grouped = df_classes.groupby(by=['serial_no']).count().reset_index()
    sno_list = list(df_classes_grouped['serial_no'][df_classes_grouped['intl_class_cd']==1])
    df_classes_subset = df_classes[df_classes['serial_no'].isin(sno_list)]

    # Remove classes - A, B and 200 
    cls_drop_list = ['A','B','200']
    df_classes_subset = df_classes_subset[~df_classes_subset['intl_class_cd'].isin(cls_drop_list)]
    df_classes_subset.reset_index(drop=True, inplace=True)
    df_classes_subset['intl_class_cd'] = df_classes_subset['intl_class_cd'].astype(int)

    print("NICE Class Shape: ", df_classes_subset.shape)
    print("NICE Class Overview: \n", df_classes_subset.head())
    print("Distinct NICE Classes: ", df_classes_subset['intl_class_cd'].unique())
    return df_classes_subset

# Get Statement Data 
def get_statement_data(df_statement, df_cf_2012): 
    print("Aggregating statement data")
    # Join texts from all statement categories.
    df_statement = pd.read_csv(statement)
    df_statement = df_statement[df_statement['serial_no'].isin(list(df_cf_2012['serial_no']))]
    df_statement['statement_text'] = df_statement['statement_text'].apply(str)
    df_statement_agg = df_statement[['statement_text','serial_no']].groupby(['serial_no'], as_index=False).agg({'statement_text':' '.join})

    print("Statement Data shape:", df_statement_agg.shape)
    print("Statement Data overview: \n", df_statement_agg.head())
    return df_statement_agg


## Merge all required data - df_cf_2012, df_classes_subset, df_statement_agg 
def merge_dfs(df_cf_2012, df_classes_subset, df_statement_agg):
    print("Merging case files, nice classes and statement data")
    dfs = [df_cf_2012[['serial_no','filing_dt','mark_id_char','acq_dist_in','acq_dist_part_in','pub_ind', 'distinct_ind']], 
        df_classes_subset[['serial_no','intl_class_cd']], df_statement_agg]

    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['serial_no'], how='inner'), dfs)
    print("Merged data shape:",df_merged.shape)
    print(df_merged.head())
    return df_merged 

'''
Pre-processing Steps  
Statement_text - 
1. lowercase 
2. remove punctuations
3. for now, remove stopwords - check if we need to remove stopwords/what kind of stopwords need to removed. 

mark_id_char - 
1. remove punctuations. 
'''

def preprocess_statement(text):
    print("Preprocessing Statement")
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    # stop_words = stopwords.words("english")
    # stopwords_dict = Counter(stop_words)
    # text_tokens = word_tokenize(text)
    # tokens_without_sw = [word for word in text_tokens if not word in stopwords_dict]
    # filtered_sentence = " ".join(tokens_without_sw)
    # return filtered_sentence 
    return text 


def preprocess_mark(text):
    print("Preprocessing Mark")
    text = re.sub(r'[^\w\s]', '', text)
    return text


if __name__ == "__main__":
    df_cf_2012, df_design, df_statement, df_classes, df_office_actions = read_data(case_file_2012_2019, statement, design_file, nice_classes, office_actions)
    df_cf_2012 = drop_null_marks(df_cf_2012)
    df_cf_2012 = drop_images(df_design, df_cf_2012)
    df_cf_2012 = drop_duplicates(df_cf_2012)
    df_cf_2012 = create_y_variable(df_cf_2012, df_office_actions)
    df_classes_subset = get_nice_classes(df_classes, df_cf_2012)
    df_statement_agg = get_statement_data(df_statement, df_cf_2012)
    df_merged = merge_dfs(df_cf_2012, df_classes_subset, df_statement_agg)
    
    # Create separate columns for unprocessed mark and statement 
    df_merged['mark_unprocessed'] = df_merged['mark_id_char'].apply(str).copy()
    df_merged['statement_unprocessed'] = df_merged['statement_text'].apply(str).copy()

    # preprocess marks 
    df_merged['mark_processed'] = df_merged['mark_id_char'].apply(lambda x: preprocess_mark(x))
    
    # preprocessing statement - only removing punctuations 
    df_merged['statement_processed'] = df_merged['statement_text'].apply(lambda x: preprocess_statement(x))
    df_merged.to_pickle(preprocessed_data)



