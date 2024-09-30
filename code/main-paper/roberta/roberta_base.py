"""
Code to create roberta-base model. Predicting distinctiveness indicator 
"""

# Import Libraries 
import sys 
sys.path.append('../../python_packages/')
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from transformers import RobertaTokenizerFast, RobertaForSequenceClassification
from time import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device available: ",device)

# Input 
data_path = '../data/roberta/df_bert_input.pkl'

# Roberta - Change for processed statement
# roberta_model_path = '../models/roberta_v1.pth'
roberta_model_path = '../models/roberta_v1_unprocessed.pth'

# Load Roberta-Base model 
model = RobertaForSequenceClassification.from_pretrained('../models/roberta_offline', num_labels=2)
tokenizer = RobertaTokenizerFast.from_pretrained('../models/roberta_ftoken_offline')
optimizer = torch.optim.Adam([
    {'params': model.parameters(), 'lr': 1e-5}
])

# Read data 
df = pd.read_pickle(data_path)
df['filing_dt'] = pd.to_datetime(df['filing_dt'])

# Incidence Rate 
print("Incidence Rate: ", ((100*df['distinct_ind'].sum())/(df.shape[0])) )

# Divide data into train, test and validation 
df_train = df[(df['filing_dt']>=pd.to_datetime('2012-01-01')) & (df['filing_dt']<=pd.to_datetime('2017-12-31'))]
df_val = df[(df['filing_dt']>=pd.to_datetime('2018-01-01')) & (df['filing_dt']<=pd.to_datetime('2018-12-31'))]
df_test = df[(df['filing_dt']>=pd.to_datetime('2019-01-01')) & (df['filing_dt']<=pd.to_datetime('2019-12-31'))] 

print("Train data shape: ", df_train.shape)
print("Validation data shape: ", df_val.shape)
print("Test data shape: ", df_test.shape)

# Create X and Y for modelling 
X_train = np.array(df_train['bert_input_unprocessed'])
y_train = np.array(df_train['distinct_ind'])
X_val = np.array(df_val['bert_input_unprocessed'])
y_val = np.array(df_val['distinct_ind'])
X_test = np.array(df_test['bert_input_unprocessed'])
y_test = np.array(df_test['distinct_ind'])

# Generate Batches 
batch_size = 16
train_max_idx = batch_size * (len(X_train)//batch_size)
val_max_idx = batch_size * (len(X_val)//batch_size)
test_max_idx = batch_size * (len(X_test)//batch_size)

X_train = X_train[:train_max_idx]
X_train = X_train.reshape(-1, batch_size)
X_train = X_train.tolist()
y_train = y_train[:train_max_idx]
y_train = y_train.reshape(-1, batch_size)

X_val = X_val[:val_max_idx]
X_val = X_val.reshape(-1, batch_size)
X_val = X_val.tolist()
y_val = y_val[:val_max_idx]
y_val = y_val.reshape(-1, batch_size)

X_test = X_test[:test_max_idx]
X_test = X_test.reshape(-1, batch_size)
X_test = X_test.tolist()
y_test = y_test[:test_max_idx]
y_test = y_test.reshape(-1, batch_size)

print("Batches Generated: ")
print("X_train: ", len(X_train))
print("y_train: ", y_train.shape)

print("X_val: ", len(X_val))
print("y_val: ", y_val.shape)

# Model Training 
from transformers.utils import logging
logging.set_verbosity(40)

from tqdm import tqdm
model = model.to(device)

num_epochs = 2
for epoch in range(num_epochs):
  model.train()
  for text, labels in tqdm(zip(X_train, y_train), total=len(X_train)):
    # prepare model input through our tokenizer
    model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    # place everything on the right device
    model_inputs = {k:v.to(device) for k,v in model_inputs.items()}
    # labels have to be torch long tensors
    labels = torch.tensor(labels).long()
    labels = labels.to(device)
    # now, we can perform the forward pass
    output = model(**model_inputs, labels=labels)
    loss, logits = output[:2]
    # and the backward pass
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

torch.save(model, roberta_model_path)
print("Model Saved")

# Predictions on validation dataset 
print("Predicting on Validation Data")
loaded_model = torch.load(roberta_model_path)
predictions, targets = [], []
pred_logits = [] 
loaded_model.eval()

with torch.no_grad():
  for text, labels in tqdm(zip(X_val, y_val), total=len(X_val)):
    try:
      model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
      model_inputs = {k:v.to(device) for k,v in model_inputs.items()}

      output = loaded_model(**model_inputs)
      logits = output[0]
      pred_logits.append(logits)
      # prediction is the argmax of the logits
      predictions.extend(logits.argmax(dim=1).tolist())
      targets.extend(labels)
    except:
      print("Unable to process: ", text)
      continue 
    
accuracy = metrics.accuracy_score(targets, predictions)
print ("accuracy", accuracy)
classification_report = metrics.classification_report(targets, predictions)
print (classification_report)


print("Predicting on Test Data")
predictions, targets = [], []
pred_logits = [] 
loaded_model.eval()

with torch.no_grad():
  for text, labels in tqdm(zip(X_test, y_test), total=len(X_test)):
    try:
      model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
      model_inputs = {k:v.to(device) for k,v in model_inputs.items()}

      output = loaded_model(**model_inputs)
      logits = output[0]
      pred_logits.append(logits)
      # prediction is the argmax of the logits
      predictions.extend(logits.argmax(dim=1).tolist())
      targets.extend(labels)
    except:
      print("Unable to process: ", text)
      continue 
    
from sklearn import metrics
accuracy = metrics.accuracy_score(targets, predictions)
print ("accuracy", accuracy)
classification_report = metrics.classification_report(targets, predictions)
print (classification_report)

