"""
Code for creating RoBERTa Input dataframe. 
<s> Mark </s> Statment_text </s> Translated </s> Present/Absent in Wordnet </s> Mark length </s> NICE Category 
</s> Pseudo Mark </s> NICE Catgory Description. 

Final DataFrame: 
serial_no, mark, statement, translation, wordnet indicator, mark length, nice category, pseudo mark, NICE cat description
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
df_mark_path = '../data/df_wordnet.pkl' 
usecols_df_mark = ['serial_no','filing_dt','mark_processed','mark_unprocessed','distinct_ind','intl_class_cd','mark_len','wn_ind']

df_statement_processed_path = '../../base_data/statement_no_pm_2012_2019.pkl'
usecols_statement_processed = ['serial_no', 'statement_processed']

df_statement_unprocessed_path = '../../base_data/statement_no_pm_unprocessed_2012_2019.pkl'
usecols_statement_unprocessed = ['serial_no', 'statement_unprocessed']

df_translated_path = '../../data/df_translated_final.pkl'
usecols_translated = ['serial_no','mark_final']

df_nice_description_path = '../data/nice_class_description.csv'

df_pm_processed_path = '../../base_data/psuedo_marks.pkl'
usecols_pm_processed = ['serial_no', 'statement_processed']

df_pm_unprocessed_path = '../../base_data/psuedo_marks_unprocessed.pkl'
usecols_pm_unprocessed = ['serial_no', 'statement_unprocessed']


# Output 
df_bert_input = '../data/roberta/df_bert_input.pkl'

"""
Steps - 
1. Check which marks are translated 
2. Preprocess NICE Class Description
3. Merge everything 
"""

class PrepareBERTData():
    def __init__(self):
        self.df_mark = pd.read_pickle(df_mark_path)[usecols_df_mark]
        print("Shape of DF Mark: ", self.df_mark.shape)
        self.df_statement_processed = pd.read_pickle(df_statement_processed_path)[usecols_statement_processed]
        print("Shape of DF Statement Processed: ", self.df_statement_processed.shape)
        self.df_statement_unprocessed = pd.read_pickle(df_statement_unprocessed_path)[usecols_statement_unprocessed]
        print("Shape of DF Statement Unprocessed: ", self.df_statement_unprocessed.shape)
        self.df_translated = pd.read_pickle(df_translated_path)[usecols_translated]
        print("Shape of DF Translated: ", self.df_translated.shape)
        self.df_nice_desc = pd.read_csv(df_nice_description_path)
        print("Shape of DF NICE Description: ", self.df_nice_desc.shape)
        self.df_pm_processed = pd.read_pickle(df_pm_processed_path)
        print("Shape of Pseudo Mark Processed: ", self.df_pm_processed.shape)
        self.df_pm_unprocessed = pd.read_pickle(df_pm_unprocessed_path)
        print("Shape of Pseudo Mark Processed: ", self.df_pm_unprocessed.shape)

    
    def add_translation(self):
        df_merged_translation = self.df_mark.merge(self.df_translated, how = 'left', on = ['serial_no'])
        assert(df_merged_translation.shape[0]==self.df_mark.shape[0])
        df_merged_translation['translated_ind'] = (df_merged_translation['mark_processed'].str.strip().str.lower() != df_merged_translation['mark_final'].str.strip().str.lower()) * 1.0 
        print("Translated Marks Percentage", (df_merged_translation['translated_ind'].sum()/df_merged_translation.shape[0]) * 100)
        df_merged_translation['mark_translated'] = np.where(df_merged_translation['translated_ind']==True, df_merged_translation['mark_final'], "no translation required")
        print("Null values:\n")
        print(df_merged_translation.isnull().sum())
        print("\n")
        return df_merged_translation

    # Preprocess NICE Description
    # 1. Lowercase 
    # 2. Remove punctuations 
    def preprocess_nice_description(self, text):
        text = str(text)
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        # stop_words = stopwords.words("english")
        # stopwords_dict = Counter(stop_words)
        # text_tokens = word_tokenize(text)
        # tokens_without_sw = [word for word in text_tokens if not word in stopwords_dict]
        # filtered_sentence = " ".join(tokens_without_sw)
        return text 


    def add_nice_description(self, df_merged_translation, preprocess=True):
        self.df_nice_desc.rename(columns = {'Class':'intl_class_cd', 'Description':'nice_description'}, inplace=True)
        if preprocess:
            self.df_nice_desc['nice_description_preprocessed'] = self.df_nice_desc['nice_description'].apply(lambda x: self.preprocess_nice_description(x))
            df_merged_nice = df_merged_translation.merge(self.df_nice_desc[['intl_class_cd','nice_description_preprocessed']], on = ['intl_class_cd'], how = 'left')
        else:
            self.df_nice_desc['nice_description_unpreprocessed'] = self.df_nice_desc['nice_description'].copy()
            df_merged_nice = df_merged_translation.merge(self.df_nice_desc[['intl_class_cd','nice_description_unpreprocessed']], on = ['intl_class_cd'], how = 'left')
        
        assert(df_merged_nice.shape[0] == df_merged_translation.shape[0])
        print("Added NICE Description")
        print("Null values: \n", df_merged_nice.isnull().sum())
        print("\n")
        return df_merged_nice 

    
    def get_pm_text(self, text):
        if text=="no Pseudo mark":
            return text 
        text = "Pseudo mark is " + text 
        return text 

    
    def add_pseudo_mark(self, df_merged_nice, preprocessed=True):
        if preprocessed:
            self.df_pm_processed.rename(columns = {'statement_processed':'pseudo_mark_processed'}, inplace=True)
            df_merged_pm = df_merged_nice.merge(self.df_pm_processed, on = ['serial_no'], how = 'left')
            df_merged_pm['pseudo_mark_processed'].fillna("no Pseudo mark", inplace=True)
            df_merged_pm['pseudo_mark_processed'] = df_merged_pm['pseudo_mark_processed'].apply(lambda x: self.get_pm_text(x))
        else:
            self.df_pm_unprocessed.rename(columns = {'statement_unprocessed':'pseudo_mark_unprocessed'}, inplace=True)
            df_merged_pm = df_merged_nice.merge(self.df_pm_unprocessed, on = ['serial_no'], how = 'left')
            df_merged_pm['pseudo_mark_unprocessed'].fillna("no Pseudo mark", inplace=True)
            df_merged_pm['pseudo_mark_unprocessed'] = df_merged_pm['pseudo_mark_unprocessed'].apply(lambda x: self.get_pm_text(x))
        
        assert(df_merged_pm.shape[0] == df_merged_nice.shape[0])
        print("Added Pseudo Marks")
        print("Null values: \n", df_merged_pm.isnull().sum())
        print("\n")
        return df_merged_pm 

    
    def add_statement_text(self, df_merged_pm, preprocessed=True):
        if preprocessed:
            df_merged = df_merged_pm.merge(self.df_statement_processed , on = ['serial_no'], how = 'left')
        else:
            df_merged = df_merged_pm.merge(self.df_statement_unprocessed , on = ['serial_no'], how = 'left')
        assert(df_merged.shape[0]==df_merged_pm.shape[0])
        print("Added statement text")
        print("Final Shape: ", df_merged.shape)
        print(df_merged.head())
        print(df_merged.columns)
        print("\n")
        return df_merged
    
    
    def add_text_ind(self, df_merged):
        df_merged['wordnet_text'] = np.where(df_merged['wn_ind']==1, 'mark present in Wordnet', 'mark absent in Wordnet')
        df_merged['mark_length_text'] = df_merged['mark_len'].apply(lambda x: "mark length is " + str(x))
        df_merged['nice_cat_text'] = df_merged['intl_class_cd'].apply(lambda x: "NICE category is " + str(x))
    

    def add_bert_input_text(self, df, preprocessed=True):
        if preprocessed:
            df['bert_input_processed'] = '<s> ' + df['mark_processed'] + ' </s> ' + df['statement_processed'] + ' </s> ' + df['mark_translated'] + ' </s> ' + df['wordnet_text'] + ' </s> ' + df['mark_length_text'] + ' </s> ' + df['nice_cat_text'] + ' </s> ' + df['nice_description_preprocessed'] + ' </s> ' + df['pseudo_mark_processed'] 
            print(df['bert_input_processed'][10])
        else:
            df['bert_input_unprocessed'] = '<s> ' + df['mark_unprocessed'] + ' </s> ' + df['statement_unprocessed'] + ' </s> ' + df['mark_translated'] + ' </s> ' + df['wordnet_text'] + ' </s> ' + df['mark_length_text'] + ' </s> ' + df['nice_cat_text'] + ' </s> ' + df['nice_description_unpreprocessed'] + ' </s> ' + df['pseudo_mark_unprocessed'] 
            print(df['bert_input_unprocessed'][10])


    def run(self):
        # Add translation 
        df_merged_translation = self.add_translation()
        
        # Add NICE Description
        df_merged_nice = self.add_nice_description(df_merged_translation)
        df_merged_nice = self.add_nice_description(df_merged_nice, preprocess=False)

        # Add Pseudo Mark 
        df_merged_pm = self.add_pseudo_mark(df_merged_nice)
        df_merged_pm = self.add_pseudo_mark(df_merged_pm, preprocessed=False)
        
        # Add Statement Text 
        df_merged_final = self.add_statement_text(df_merged_pm)
        df_merged_final = self.add_statement_text(df_merged_final, preprocessed=False)
        df_merged_final.dropna(inplace=True)
        df_merged_final.reset_index(drop=True, inplace=True)
        
        # Replace indicators with text 
        self.add_text_ind(df_merged_final)
        
        # Add BERT Input 
        self.add_bert_input_text(df_merged_final)
        self.add_bert_input_text(df_merged_final, preprocessed=False)

        print("Final Shape: ", df_merged_final.shape)
        print("Null values: \n", df_merged_final.isnull().sum())
        print("Dumping File")
        
        df_merged_final.to_pickle(df_bert_input)
        print(df_merged_final.head())
        return df_merged_final 


if __name__ == "__main__":
    prepare_bert_data = PrepareBERTData()
    prepare_bert_data.run()
