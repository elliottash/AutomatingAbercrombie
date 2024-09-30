import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error, roc_auc_score, roc_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pickle
import matplotlib.pyplot as plt
from time import time 
seed = 42
np.random.seed(seed)


# Input 
xgb_model_path = '../models/xgb_ft_diff.pkl'
x_val_path = '../data/xgboost/X_val.pkl'
x_test_path = '../data/xgnoost/X_test.pkl'
y_val_path = '../data/xgnoost/y_train.pkl'
y_test_path = '../data/xgboost/y_test.pkl'


# Load data and model 

# XGB Model
with open(xgb_model_path, 'rb') as pickle_file:
    xgb_model = pickle.load(pickle_file)



x_val = pd.read_pickle(x_val_path)
x_test = pd.read_pickle(x_test_path)
y_val = pd.read_pickle(y_val_path)['distinct_ind'].tolist()
y_test = pd.read_pickle(y_test_path)['distinct_ind'].tolist()

# Function to predict model scores
def predict_scores(model, x, th=0.5):
    pred_probs = model.predict_proba(x)
    predictions = np.array([1 if x>=th else 0 for x in pred_probs[:,1]])
    return pred_probs[:,1], predictions 

# Function to get threshold - max difference b/w tpr and fpr 
def get_threshold(val_targets, val_pred):
    fpr_val, tpr_val, threshold_val = roc_curve(val_targets, val_pred)
    th_val = threshold_val[np.argmax(tpr_val-fpr_val)]
    return th_val

# Function to evaluate predictions - classification report 
def classification_metrics(targets, predictions):
    accuracy = metrics.accuracy_score(targets, predictions)
    print ("accuracy", accuracy)
    classification_report = metrics.classification_report(targets, predictions)
    print (classification_report)
    return accuracy, classification_report

# Function to evaluate AUC ROC scores 
def auc_roc(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return auc, fpr, tpr

# Function to plot ROC Curve 
def plot_auc_roc(fpr, tpr, roc_auc, save_loc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(save_loc)
    plt.close()

# Function to generate decile chart over probabilities  
def decile_chart(df, pred_prob, pred, targets, cuts=10):
    # Sort by predicted probabilites 
    df.sort_values(by=pred_prob, ascending=False, inplace=True)
    # Split df into subsets 
    df_split = np.array_split(df, cuts)
    # Create output dataframe - (#cuts, accuracy, precision, recall, f1) 
    df_output = pd.DataFrame(columns=['n','n_0','n_1','pred_0','pred_1','acc','prec','recall','f1'])
    n_list = []
    n_0_list = [] 
    n_1_list = []
    pred_0_list = []
    pred_1_list = []
    acc_list = [] 
    prec_list = [] 
    recall_list = [] 
    f1_list = [] 
    for i in range(cuts):
        df_to_process = df_split[i]
        n = df_to_process.shape[0]
        n_1 = df_to_process[targets].sum()
        n_0 = n - n_1 
        pred_1 = df_to_process[pred].sum()
        pred_0 = n - pred_1
        tn, fp, fn, tp = confusion_matrix(df_to_process[targets].tolist(), df_to_process[pred].tolist(), labels=[0, 1]).ravel()
        acc = np.round(100*(tp + tn)/(tn+fp+fn+tp),2)
        prec = np.round(100*(tp)/(fp+tp),2)
        recall = np.round(100*(tp)/(fn+tp),2)
        f1 = np.round((2*prec*recall)/(prec+recall))
        
        n_list.append(n)
        n_1_list.append(n_1)
        n_0_list.append(n_0)
        pred_0_list.append(pred_0)
        pred_1_list.append(pred_1)
        acc_list.append(acc)
        prec_list.append(prec)
        recall_list.append(recall)
        f1_list.append(f1)
    df_output['n'] = n_list
    df_output['n_0'] = n_0_list
    df_output['n_1'] = n_1_list
    df_output['pred_0'] = pred_0_list
    df_output['pred_1'] = pred_1_list
    df_output['acc'] = acc_list
    df_output['prec'] = prec_list
    df_output['recall'] = recall_list
    df_output['f1'] = f1_list
    return df_output 

# Function to get feature importance 
def get_feature_importance(xgb_model):
    feature_important = xgb_model.get_booster().get_score(importance_type='weight')
    keys = list(feature_important.keys())
    values = list(feature_important.values())
    data = pd.DataFrame(data=values, index=keys, columns=["score"]).sort_values(by = "score", ascending=False)
    print("Feature Importance: ")
    print(data.nlargest(10, columns="score"))



# Feature Importance 
get_feature_importance(xgb_model)


# Test Dataset 
pred_prob_test, predictions_test = predict_scores(xgb_model, x_test)
acc1, classification_report1 = classification_metrics(y_test, predictions_test)
print("Accuracy:\n", acc1)
auc1, fpr, tpr = auc_roc(y_test, pred_prob_test)
print("AUC:\n", auc1)
plot_auc_roc(fpr, tpr, auc1, '../plots/auc_xgb.png')
# Decile Chart 
df_decile_input = pd.DataFrame(columns = ['pred_prob','pred','targets'])
df_decile_input['pred_prob'] = pred_prob_test
df_decile_input['pred'] = predictions_test
df_decile_input['targets'] = y_test
print(df_decile_input.head(10))
decile1 = decile_chart(df_decile_input, 'pred_prob', 'pred', 'targets', cuts=10)
decile1.to_csv('../plots/xgb_decile1.csv')
print("Decile Report:\n", decile1)
print("\n")
