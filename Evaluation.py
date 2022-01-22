import random
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, \
    precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

warnings.filterwarnings('ignore')


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
    n_classes = y_test_bin.shape[1]

    return y_test_bin, np.array(list(scores)), n_classes


def ROC_evaluation(predictions):
    true, scores, n_classes = preproc(predictions)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(true[:, i], scores[:, i])
        plt.plot(fpr[i], tpr[i], color='darkorange', lw=2)
        print('AUC for Class {}: {}'.format(i + 1, auc(fpr[i], tpr[i])))

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        color="navy",
        linestyle=":",
        linewidth=4,
    )
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.savefig("plot/ROC_curve.svg", dpi=1200)
    plt.clf()


def evaluation(predictions):
    x = pd.read_csv(predictions)
    y_true = x.REAL
    y_pred = x.PREDICTED
    labels = np.unique(y_true)

    precision = {}
    recall = {}
    average_precision = {}

    confusion = confusion_matrix(y_true, y_pred, labels=labels)
    # print("CONFUSION MATRIX\n", confusion)
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_true, y_pred)))
    confusion_df = pd.DataFrame(confusion, index=labels, columns=labels)
    confusion_df_sample = confusion_df.sample(frac=0.3, replace=True, random_state=1)
    sns.heatmap(confusion_df_sample, annot=False).get_figure().savefig('plot/cnf_matrix_sample.svg',
                                                                       bbox_inches="tight")
    sns.heatmap(confusion_df, annot=False).get_figure().savefig('plot/cnf_matrix.svg', bbox_inches="tight")
    confusion_df.transpose().to_csv("Confusion Matrix.csv")
    plt.clf()

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
    print('Weighted F1-score: {:.2f}'.format(average_precision["weighted"]))

    # CLASSIFICATION REPORT
    clf_report = classification_report(y_true, y_pred, output_dict=True)
    clf_report_sample = sample_from_dict(clf_report, sample=50)
    # print('\nClassification Report\n')
    # print(clf_report)
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=False).get_figure().savefig('plot/clf_report.svg',
                                                                                           bbox_inches="tight")
    plt.clf()
    sns.heatmap(pd.DataFrame(clf_report_sample).iloc[:-1, :].T, annot=False).get_figure().savefig(
        'plot/clf_report_sample.svg', bbox_inches="tight")
    plt.clf()
    pd.DataFrame(clf_report).transpose().to_csv("Classification Report.csv")

    ROC_evaluation(predictions)


def sample_from_dict(d, sample=10):
    keys = random.sample(list(d), sample)
    values = [d[k] for k in keys]
    return dict(zip(keys, values))
