'''
Code to run xgboost model with fasttext embedding difference of mark_id_char and statement
along with nice classes, mark_length and wordnet indicator.
Dependent variable = distinct_ind
'''


## Import libraries 
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
import pickle
from time import time 
seed = 42
np.random.seed(seed)

## Input 
modelling_dataset = '../data/distilbert/df_bert_input.pkl'
modelling_dataset_ft = '../../reg_indicator/data/df_ft_diff.pkl'

## Output 
X_train_path = '../data/X_train.pkl'
X_val_path = '../data/X_val.pkl'
X_test_path = '../data/X_test.pkl'
y_train_path = '../data/y_train.pkl'
y_val_path = '../data/y_val.pkl'
y_test_path = '../data/y_test.pkl'
xgb_model_location = '../models/xgb_ft_diff.pkl'

## Script variables 
y = 'distinct_ind'
key = ['serial_no','filing_dt']


## Read files
req_cols_base = ['serial_no','filing_dt','distinct_ind','intl_class_cd','mark_len','wn_ind']
df_ft = pd.read_pickle(modelling_dataset_ft)
df_model_base = pd.read_pickle(modelling_dataset)[req_cols_base]
print("Shape of modeling base: ", df_model_base.shape)
print("Shape of Fasttext Embeddings: ", df_ft.shape)

# One hot encode intl_class_cd
one_hot = pd.get_dummies(df_model_base['intl_class_cd'])
cols = list(one_hot.columns)
one_hot_columns = ['nice_class_' + str(x) for x in cols]
one_hot.columns = one_hot_columns 
df_model_base = df_model_base.join(one_hot)
df_model_base.drop(columns=['intl_class_cd'],inplace=True)
print("Shape after one-hot encoding:",df_model_base.shape)
print("Data sample:\n")
print(df_model_base.head())


# Merge with fasttext embedding difference
df_merged = df_model_base.merge(df_ft, on = ['serial_no'], how = 'inner')
print("Shape after merging data:", df_merged.shape)


# Split intro train, validation and test datasets 
# train = 2012-2017 
# validation = 2018
# test = 2019 
df_merged['filing_dt'] = pd.to_datetime(df_merged['filing_dt'])
df_train = df_merged[(df_merged['filing_dt']>=pd.to_datetime('2012-01-01')) & (df_merged['filing_dt']<=pd.to_datetime('2017-01-31'))]
df_val = df_merged[(df_merged['filing_dt']>=pd.to_datetime('2018-01-01')) & (df_merged['filing_dt']<=pd.to_datetime('2018-12-31'))]
df_test = df_merged[(df_merged['filing_dt']>=pd.to_datetime('2019-01-01'))]
print("Shape of training data: ", df_train.shape)
print("Shape of validation data: ", df_val.shape)
print("Shape of test data: ", df_test.shape)


# X and Y variables
X_train = df_train.drop(columns = key+[y])
y_train = df_train[[y]]

X_val = df_val.drop(columns = key+[y])
y_val = df_val[[y]]

X_test = df_test.drop(columns = key+[y])
y_test = df_test[[y]]

# Save train, val and test files 
datasets = [X_train, X_val, X_test, y_train, y_val, y_test]
paths = [X_train_path, X_val_path, X_test_path, y_train_path, y_val_path, y_test_path]
for i in range(len(datasets)):
    pickle.dump(datasets[i], open(paths[i], "wb"))


# Xgboost Model 
start = time()
print("Training Started")
xgb_model = xgb.XGBClassifier(n_estimators=500, max_depth = 7, learning_rate = 0.05, 
                              objective="binary:logistic", random_state=seed)

xgb_model.fit(X_train, y_train, verbose=True)
end = time()
print("Time taken: " + str((end-start)/60) + " min") 

# dump model 
pickle.dump(xgb_model, open(xgb_model_location, "wb"))
