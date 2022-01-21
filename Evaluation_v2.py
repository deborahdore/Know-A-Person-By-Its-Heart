import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, \
    precision_score, recall_score, f1_score, PrecisionRecallDisplay, ConfusionMatrixDisplay
import pandas as pd
import numpy as np
import warnings
import random
warnings.filterwarnings('ignore')

def Evaluation(predictions):
    predictions = "datasets/predictions_dataset.csv"
    x = pd.read_csv(predictions)
    y_true = x.REAL
    y_pred = x.PREDICTED
    labels = np.unique(y_true)

    precision={}
    recall={}
    average_precision={}

    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    #print("CONFUSION MATRIX\n", confusion)
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_true, y_pred)))
    confusion_df = pd.DataFrame(confusion, index=labels, columns=labels)
    confusion_df_sample = confusion_df.sample(frac=0.3, replace=True, random_state=1)
    sns.heatmap(confusion_df_sample, annot=False).get_figure().savefig('plot/cnf_matrix_sample.png', bbox_inches="tight")
    sns.heatmap(confusion_df, annot=False).get_figure().savefig('plot/cnf_matrix.png', bbox_inches="tight")
    confusion_df.transpose().to_csv("Confusion Matrix.csv")

    #MICRO
    precision["micro"] = precision_score(y_true, y_pred, average='micro')
    recall["micro"] = recall_score(y_true, y_pred, average='micro')
    average_precision["micro"] = f1_score(y_true, y_pred, average='micro')
    print('Micro Precision: {:.2f}'.format(precision["micro"]))
    print('Micro Recall: {:.2f}'.format(recall["micro"]))
    print('Micro F1-score: {:.2f}\n'.format(average_precision["micro"]))


    #MACRO
    precision["macro"] = precision_score(y_true, y_pred, average='macro')
    recall["macro"] = recall_score(y_true, y_pred, average='macro')
    average_precision["macro"] = f1_score(y_true, y_pred, average='macro')
    print('Macro Precision: {:.2f}'.format(precision["macro"]))
    print('Macro Recall: {:.2f}'.format(recall["macro"]))
    print('Macro F1-score: {:.2f}\n'.format(average_precision["macro"]))

    #WEIGHTED
    precision["weighted"] = precision_score(y_true, y_pred, average='weighted')
    recall["weighted"] = recall_score(y_true, y_pred, average='weighted')
    average_precision["weighted"] = f1_score(y_true, y_pred, average='weighted')
    print('Weighted Precision: {:.2f}'.format(precision["weighted"]))
    print('Weighted Recall: {:.2f}'.format(recall["weighted"]))
    print('Weighted F1-score: {:.2f}'.format(average_precision["weighted"]))

    #CLASSIFICATION REPORT
    clf_report = classification_report(y_true, y_pred, output_dict=True)
    clf_report_sample = sample_from_dict(clf_report, sample=50)
    #print('\nClassification Report\n')
    #print(clf_report)
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=False).get_figure().savefig('plot/clf_report.png', bbox_inches="tight")
    sns.heatmap(pd.DataFrame(clf_report_sample).iloc[:-1, :].T, annot=False).get_figure().savefig('plot/clf_report_sample.png', bbox_inches="tight")
    pd.DataFrame(clf_report).transpose().to_csv("Classification Report.csv")


def sample_from_dict(d, sample=10):
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))