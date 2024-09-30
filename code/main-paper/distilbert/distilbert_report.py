"""
Get DistilBERT Report 
"""

import pandas as pd
import numpy as np
import time
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from time import time
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
plt.rcParams["figure.dpi"] = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device available: ",device)
seed = 42
np.random.seed(seed)

# Input
# Trained Model
distilbert_model = 'models/distilbert_unprocessed_v1.pth'
# Input Data
data_path = '../../data/distilbert/df_bert_input.pkl'
# DistilBERT Tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-cased')

# Function to generate batches and reshape
def batching(np_array, max_idx, batch_size):
    np_array = np_array[:max_idx]
    np_array = np_array.reshape(-1, batch_size)
    batched_list = np_array.tolist()
    return batched_list

# Temperature Scaling
def T_scaling(logits, args):
    temperature = args.get('temperature', None)
    return torch.div(logits, temperature)

# Function for predicting y
def bert_pred(model_path, X, y, th=0.5, temp=None):
    loaded_model = torch.load(model_path)
    predictions, targets = [], []
    pred_prob = []
    loaded_model.eval()
    with torch.no_grad():
        for text, labels in tqdm(zip(X, y), total=len(X)):
            #try:
            model_inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            model_inputs = {k:v.to(device) for k,v in model_inputs.items()}
            output = loaded_model(**model_inputs)
            logits = output[0]
            if temp is not None:
                logits = T_scaling(logits, {'temperature':temp})
            probs = torch.sigmoid(logits)
            pred_prob.extend(probs[:,1].tolist())
            if th==0.5:
                # prediction is the argmax of the logits
                predictions.extend(logits.argmax(dim=1).tolist())
            else:
                predictions.extend(((probs[:,1]>=th)*1).tolist())
            targets.extend(labels)
            #except:
            #    print("Unable to process: ", text)
            #    continue
    return targets, pred_prob, predictions

# Function to evaluate predictions - classification report
def classification_metrics(targets, predictions, conf_matrix_loc):
    accuracy = metrics.accuracy_score(targets, predictions)
    print ("accuracy", accuracy)
    classification_report = metrics.classification_report(targets, predictions)
    print (classification_report)
    conf_matrix = confusion_matrix(targets, predictions)
    # Save confusion matrix
    ax = sns.heatmap(conf_matrix, annot=True, cmap='Blues')
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.savefig(conf_matrix_loc)
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

# Function to get threshold  - max difference b/w tpr and fpr
def get_threshold(val_targets, val_pred):
    fpr_val, tpr_val, threshold_val = roc_curve(val_targets, val_pred)
    th_val = threshold_val[np.argmax(tpr_val-fpr_val)]
    return th_val

# Function to get calibration plots - 10 bins
def get_calibration_curve10(target, pred_prob, save_loc, nbins=10):
    plot_y, plot_x = calibration_curve(target, pred_prob, n_bins=nbins)
    # calibration curves
    fig, ax = plt.subplots()
    plt.plot(plot_y, plot_x, marker='o', linewidth=1, label='logreg')

    # reference line, legends, and axis labels
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    #ax.hist(pred_prob, weights=np.ones(len(pred_prob)) / len(pred_prob), bins=nbins, color='papayawhip')
    fig.suptitle('Calibration plot for DistilBERT')
    ax.set_xlabel('Mean Predicted probability')
    ax.set_ylabel('True probability in each bin')
    plt.savefig(save_loc)
    plt.close()

    # Get Histogram
    fig, ax = plt.subplots()
    ax.hist(pred_prob, weights=np.ones(len(pred_prob)) / len(pred_prob), bins=nbins, color='green')
    fig.suptitle('Histogram for probability distribution')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Counts')
    plt.savefig(save_loc[:-4] + '_hist.png')
    plt.close()

# Function to get calibration plots - 10 bins and overlayed
def get_calibration_curve10_overlay(target, pred_prob, save_loc, nbins=10):
    plot_y, plot_x = calibration_curve(target, pred_prob, n_bins=nbins)
    # calibration curves
    #fig, ax = plt.subplots()
    plt.plot(plot_y, plot_x, marker='o', linewidth=1, label='logreg')
    # reference line, legends, and axis labels
    ref_x = np.arange(0, 1.1, 0.1)
    ref_y = np.arange(0, 1.1, 0.1)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlim(left=0)
    plt.plot(ref_x, ref_y, color='black', linestyle='dashed')
    # Histogram
    plt.hist(pred_prob, weights=np.ones(len(pred_prob)) / len(pred_prob), bins=nbins, color='green')
    # Title and labels
    plt.xlabel('Mean Predicted probability')
    plt.ylabel('True probability in each bin')
    plt.title('Calibration plot for DistilBERT')
    plt.savefig(save_loc)
    plt.close()

# Function to get precision at different thresholds
def get_precision_at_thresholds(target, pred_prob, save_loc, thresholds=np.arange(0.1,1,0.1)):
    prec_dict = {'threshold':[], 'prec0':[], 'prec1':[], 'n0_actual':[], 'n1_actual':[], 'n0_pred':[], 'n1_pred':[], 'n_total':[]}
    for th in thresholds:
        predictions = [1 if x>=th else 0 for x in pred_prob]
        precisions = precision_score(target, predictions, average=None)
        prec0 = precisions[0]
        prec1 = precisions[1]
        n_total = len(target)
        n1_actual = sum(target)
        n0_actual = n_total - n1_actual
        n1_pred = sum(predictions)
        n0_pred = n_total - n1_pred
        prec_dict['threshold'].append(th)
        prec_dict['prec0'].append(prec0)
        prec_dict['prec1'].append(prec1)
        prec_dict['n0_actual'].append(n0_actual)
        prec_dict['n1_actual'].append(n1_actual)
        prec_dict['n0_pred'].append(n0_pred)
        prec_dict['n1_pred'].append(n1_pred)
        prec_dict['n_total'].append(n_total)
    df_prec = pd.DataFrame(prec_dict)
    df_prec.to_csv(save_loc, index=False)

# Function to convert pred probabilities into logits
def prob_to_logits(pred_prob):
    probabilities = np.array(pred_prob, dtype=np.float64)
    logits_array = np.log(probabilities/(1-probabilities))
    return logits_array

# Function to calibrate probabilities using temperature scaling
def calibrate_classifier(pred_prob, labels):
    temperature = nn.Parameter(torch.ones(1).cuda())
    args = {'temperature': temperature}
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.LBFGS([temperature], lr=0.001, max_iter=10000, line_search_fn='strong_wolfe')
    logits_array = prob_to_logits(pred_prob)
    labels_array = np.array(labels, dtype=np.float64)
    temps = []
    losses = []

    # Create tensors
    logits_list = torch.from_numpy(logits_array).to(device)
    labels_list = torch.from_numpy(labels_array).to(device)

    # Optimize for temperature
    def _eval():
        loss = criterion(T_scaling(logits_list, args), labels_list)
        loss.backward()
        temps.append(temperature.item())
        losses.append(loss)
        #print("Loss-List: ", loss)
        #print("Temperature-List: ", temps)
        return loss

    optimizer.step(_eval)
    print('Final T_scaling factor: {:.2f}'.format(temperature.item()))
    return temperature.item()


def report_metrics(op_path, temperature=None):
    # Prediction on test data
    if temperature is None:
        labels, pred_prob, predictions = bert_pred(distilbert_model, X_test, y_test)
    else:
        labels, pred_prob, predictions = bert_pred(distilbert_model, X_test, y_test, temp=temperature)

    # Classification Metrics
    acc_model, classification_report_model = classification_metrics(labels, predictions, op_path['conf_matrix_loc'])
    print("Accuracy: ", acc_model)
    print("\n")
    print("Classification Report: ", classification_report_model)
    print("\n")

    # ROC Score
    auc_model, fpr, tpr = auc_roc(labels, pred_prob)
    print("AUC: ", auc_model)
    print("\n")

    # Plot ROC
    plot_auc_roc(fpr, tpr, auc_model, op_path['roc_auc_path'])

    # Calibration Plot
    get_calibration_curve10(labels, pred_prob, op_path['calib_plot_path'])
    get_calibration_curve10_overlay(labels, pred_prob, op_path['calib_plot_path_overlay'])

    # Precision at different thresholds
    get_precision_at_thresholds(labels, pred_prob, op_path['prec_at_th_path'])

    # Decile Chart
    df_decile_input = pd.DataFrame(columns = ['pred_prob','pred','targets'])
    df_decile_input['pred_prob'] = pred_prob
    df_decile_input['pred'] = predictions
    df_decile_input['targets'] = labels
    decile = decile_chart(df_decile_input, 'pred_prob', 'pred', 'targets', cuts=10)
    decile.to_csv(op_path['decile_path'])
    print("Decile Report: ", decile)
    print("\n")

    return labels, pred_prob, predictions


# Read data
df = pd.read_pickle(data_path)
df['filing_dt'] = pd.to_datetime(df['filing_dt'])


# Divide data into train, test and validation
df_train = df[(df['filing_dt']>=pd.to_datetime('2012-01-01')) & (df['filing_dt']<=pd.to_datetime('2017-12-31'))]
df_val = df[(df['filing_dt']>=pd.to_datetime('2018-01-01')) & (df['filing_dt']<=pd.to_datetime('2018-12-31'))]
df_test = df[(df['filing_dt']>=pd.to_datetime('2019-01-01')) & (df['filing_dt']<=pd.to_datetime('2019-12-31'))]
print("Train data shape: ", df_train.shape)
print("Validation data shape: ", df_val.shape)
print("Test data shape: ", df_test.shape)

# X_test
X_test = np.array(df_test['bert_input_unprocessed'])
# y_test
y_test = np.array(df_test['distinct_ind'])

batch_size = 16
# Batching
test_max_idx = batch_size * (len(X_test)//batch_size)
X_test = batching(X_test, test_max_idx, batch_size=16)
y_test = batching(y_test, test_max_idx, batch_size=16)

print("Data prepared successfully")

# Output Path Dictionary
op_no_temp = {'roc_auc_path':'plots/auc_dbert.png',
              'calib_plot_path':'plots/calib_plot_dbert.png',
              'calib_plot_path_overlay':'plots/calib_plot_dbert_overlay10.png',
              'prec_at_th_path':'plots/threshold_precisions.csv',
              'decile_path':'plots/decile_bert.csv',
              'conf_matrix_loc':'plots/confusion_matrix.png'}

# Get metrics
labels, pred_prob, predictions = report_metrics(op_no_temp)
# Get Optimal Temperature
temperature = calibrate_classifier(pred_prob, labels)

# Output Path Dictionary - With T_scaling
op_temp = {'roc_auc_path':'plots/auc_dbert_tscaled.png',
          'calib_plot_path':'plots/calib_plot_dbert_tscaled.png',
          'calib_plot_path_overlay':'plots/calib_plot_dbert_overlay10_tscaled.png',
          'prec_at_th_path':'plots/threshold_precisions_tscaled.csv',
          'decile_path':'plots/decile_bert_tscaled.csv',
          'conf_matrix_loc':'plots/confusion_matrix_tscaled.png'}

# Get New Metrics - with temperature scaling
if np.isnan(temperature) == False:
    report_metrics(op_temp, temperature=temperature)
else:
    print("Temperature is NaN")

