
import random
import warnings

import numpy as np
import pandas as pd

import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, \
    precision_score, recall_score, f1_score, roc_curve, auc, cohen_kappa_score, matthews_corrcoef, roc_auc_score
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')

def evaluation(predictions):
    x = pd.read_csv(predictions)
    y_true, y_pred, y_true_bin, y_scores, labels = preproc(predictions)

    Confusion_matrix(y_true, y_pred, labels)
    Precision_Recall(y_true, y_pred)
    Classification_report(y_true, y_pred)
    ROC_evaluation(y_true_bin, y_scores, labels)
    print("----OTHER SCORES----")
    print("Matthews correlation coefficient:", round(matthews_corrcoef(y_true, y_pred), 2))
    print("Cohen Kappa score:", round(cohen_kappa_score(y_true, y_pred), 2))


def Confusion_matrix(y_true, y_pred, labels):
    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    print("---CONFUSION MATRIX METHOD---")
    print('Accuracy: {:.2f}'.format(accuracy_score(y_true, y_pred)))
    print("complete confusion matrix in Confusion_Matrix.csv\n")
    confusion_df = pd.DataFrame(confusion, index=labels, columns=labels)
    confusion_df_sample = confusion_df.sample(frac=0.3, replace=True, random_state=1)
    sns.heatmap(confusion_df_sample, annot=False).get_figure().savefig('plot/cnf_matrix_sample.svg',
                                                                       bbox_inches="tight")
    sns.heatmap(confusion_df, annot=False).get_figure().savefig('plot/cnf_matrix.svg', bbox_inches="tight")
    confusion_df.transpose().to_csv("Confusion_Matrix.csv")
    plt.clf()

def Precision_Recall(y_true, y_pred):
    precision = {}
    recall = {}
    average_precision = {}
    print("---PRECISON RECALL METHOD---")
    # MICRO
    precision["micro"] = precision_score(y_true, y_pred, average='micro')
    recall["micro"] = recall_score(y_true, y_pred, average='micro')
    average_precision["micro"] = f1_score(y_true, y_pred, average='micro')
    print('Micro Precision: {:.2f}'.format(precision["micro"]))
    print('Micro Recall: {:.2f}'.format(recall["micro"]))
    print('Micro F1-score: {:.2f}\n'.format(average_precision["micro"]))

    # MACRO
    precision["macro"] = precision_score(y_true, y_pred, average='macro')
    recall["macro"] = recall_score(y_true, y_pred, average='macro')
    average_precision["macro"] = f1_score(y_true, y_pred, average='macro')
    print('Macro Precision: {:.2f}'.format(precision["macro"]))
    print('Macro Recall: {:.2f}'.format(recall["macro"]))
    print('Macro F1-score: {:.2f}\n'.format(average_precision["macro"]))

    # WEIGHTED
    precision["weighted"] = precision_score(y_true, y_pred, average='weighted')
    recall["weighted"] = recall_score(y_true, y_pred, average='weighted')
    average_precision["weighted"] = f1_score(y_true, y_pred, average='weighted')
    print('Weighted Precision: {:.2f}'.format(precision["weighted"]))
    print('Weighted Recall: {:.2f}'.format(recall["weighted"]))
    print('Weighted F1-score: {:.2f}\n'.format(average_precision["weighted"]))

def Classification_report(y_true, y_pred):
    clf_report = classification_report(y_true, y_pred, output_dict=True)
    print("---CLASSIFICATION REPORT METHOD---")
    print("complete classification report in Classification_Report.csv\n")

    clf_report_sample = sample_from_dict(clf_report, sample=50)
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=False).get_figure().savefig('plot/clf_report.svg',
                                                                                           bbox_inches="tight")
    plt.clf()
    sns.heatmap(pd.DataFrame(clf_report_sample).iloc[:-1, :].T, annot=False).get_figure().savefig(
        'plot/clf_report_sample.svg', bbox_inches="tight")
    plt.clf()

    pd.DataFrame(clf_report).transpose().to_csv("Classification_Report.csv")

def ROC_evaluation(y_true_bin, y_scores, labels):
    roc_auc = {}
    roc_sum =  0
    for i in range(len(labels)):
        roc_score = round(roc_auc_score(y_true_bin[:,i], y_scores[:,i], average="weighted", multi_class="ovr"),2)
        roc_auc[labels[i]] = roc_score
        roc_sum += roc_score
        #print('AUC ROC for Class {}: {}'.format(labels[i], round(roc_score, 2)))

    pd.DataFrame(roc_auc.items(), columns=['Patient','AUC ROC score']).to_csv("AUC_ROC_score.csv")

    print("----ROC METHOD----")
    print("Mean AUC ROC: ", round(roc_sum/len(labels), 2))
    print("score for each class in AUC_ROC_score.csv\n")

def preproc(predictions):
    x = pd.read_csv(predictions)
    real = x.REAL
    pred = x.PREDICTED
    scores = x.SCORES
    for i in range(len(scores)):
        scores[i] = scores[i][1:-1]
        scores[i] = scores[i].split()
        for j in range(len(scores[i])):
            scores[i][j] = float(scores[i][j])
    y_test_bin = label_binarize(real, classes=np.unique(real))
    classes = np.unique(real)

    return real, pred, y_test_bin, np.array(list(scores)), classes

def sample_from_dict(d, sample):
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))
