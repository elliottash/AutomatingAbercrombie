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
import seaborn as sns
from time import time 
seed = 42
np.random.seed(seed)
from time import gmtime, strftime


# Input Paths 
xgb_model_path = '../models/xgb_ft_diff.pkl'
x_val_path = '../data/X_val.pkl'
x_test_path = '../data/X_test.pkl'
y_val_path = '../data/y_train.pkl'
y_test_path = '../data/y_test.pkl'


# Output Paths 
## Output
curr_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())

acc_list_path = 'xgboost_boot/accuracy_list_' + curr_time + '.pickle'
auc_list_path = 'xgboost_boot/auc_list_' + curr_time + '.pickle'
prec_list_path = 'xgboost_boot/prec_list_' + curr_time + '.pickle'
recall_list_path = 'xgboost_boot/recall_list_' + curr_time + '.pickle'
f1_list_path = 'xgboost_boot/f1_list_' + curr_time + '.pickle'
results_path = 'xgboost_boot/df_bootstrap_results.csv'
summary_stats_path = 'xgboost_boot/summary_stats.csv'
auc_plot_path = 'xgboost_boot/auc_plot.png'
cls_plot_path = 'xgboost_boot/cls_plot.png'

## Load Model and Dataset 
def load_stuff():
    # Xgb model 
    with open(xgb_model_path, 'rb') as pickle_file:
        xgb_model = pickle.load(pickle_file)
    
    # Test Set 
    x_test = pd.read_pickle(x_test_path)
    y_test = pd.read_pickle(y_test_path)['distinct_ind'].tolist()

    return xgb_model, x_test, y_test 


## Function to evaluate predictions - classification report
def classification_metrics(targets, predictions, pred_prob):
    accuracy = metrics.accuracy_score(targets, predictions)
    auc_score = roc_auc_score(targets, pred_prob)
    cls_report = metrics.classification_report(targets, predictions, output_dict=True)
    precision = cls_report['weighted avg']['precision']
    recall = cls_report['weighted avg']['recall']
    f1 = cls_report['weighted avg']['f1-score']
    return accuracy, auc_score, precision, recall, f1

## Function to evaluate AUC ROC scores
def auc_roc(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return auc, fpr, tpr

# Function to predict model scores
def predict_scores(model, x, th=0.5):
    pred_probs = model.predict_proba(x)
    predictions = [0]*len(pred_probs[:,1])
    for i in range(len(predictions)):
        if pred_probs[i][1]>=th:
            predictions[i] = 1
        
    return pred_probs[:,1], predictions 


## Bootstrap Functions  
def bootstrap(nrounds, x_test, y_test, xgb_model):
    acc_list = []
    auc_list = []
    prec_list = []
    recall_list = []
    f1_list = []
    df_test = x_test.copy()
    xcols = list(x_test.columns)
    df_test['distinct_ind'] = y_test 

    for i in range(nrounds):
        # Sample with replacements
        df_test = df_test.sample(frac=1, replace=True)
        X_test, y_test = df_test[xcols], df_test['distinct_ind']
        labels = y_test.tolist()
        pred_prob, predictions = predict_scores(xgb_model, X_test)
        acc, auc, prec, recall, f1 = classification_metrics(labels, predictions, pred_prob)
        print("Round {round}: accuracy {accr}".format(round=i+1, accr=acc))
        acc_list.append(acc)
        auc_list.append(auc)
        prec_list.append(prec)
        recall_list.append(recall)
        f1_list.append(f1)

    return acc_list, auc_list, prec_list, recall_list, f1_list

def dump_files(acc_list, auc_list, prec_list, recall_list, f1_list):
    # Dump
    with open(acc_list_path, 'wb') as handle:
        pickle.dump(acc_list, handle)

    with open(auc_list_path, 'wb') as handle:
        pickle.dump(auc_list, handle)

    with open(prec_list_path, 'wb') as handle:
        pickle.dump(prec_list, handle)

    with open(recall_list_path, 'wb') as handle:
        pickle.dump(recall_list, handle)

    with open(f1_list_path, 'wb') as handle:
        pickle.dump(f1_list, handle)

    df_bootstrap_results = pd.DataFrame({'Accuracy':acc_list, 'AUC':auc_list, 'Precision':prec_list,
                        'Recall':recall_list, 'F1':f1_list})
    df_bootstrap_results.to_csv(results_path, index=False)
    df_bootstrap_results.describe().to_csv(summary_stats_path)
    return df_bootstrap_results

def get_plots(df_bootstrap_results, auc_plot_path=auc_plot_path, cls_plot_path=cls_plot_path):
    sns.violinplot(data=df_bootstrap_results[['Accuracy','Precision','Recall','F1']])
    plt.savefig(cls_plot_path)
    plt.close()
    sns.violinplot(data=df_bootstrap_results[['AUC']])
    plt.savefig(auc_plot_path)
    plt.close()

if __name__ == "__main__":
    start = time()
    xgb_model, x_test, y_test = load_stuff()
    acc_list, auc_list, prec_list, recall_list, f1_list = bootstrap(30, x_test, y_test, xgb_model)
    df_bootstrap_results = dump_files(acc_list, auc_list, prec_list, recall_list, f1_list)
    get_plots(df_bootstrap_results)
    end = time()
    print("Time taken: ", np.round((end-start)/60,2))