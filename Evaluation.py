import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import label_binarize


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
        fpr[i], tpr[i], _ = metrics.roc_curve(true[:, i], scores[:, i])
        plt.plot(fpr[i], tpr[i], color='darkorange', lw=2)
        print('AUC for Class {}: {}'.format(i + 1, metrics.auc(fpr[i], tpr[i])))

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = metrics.auc(fpr["macro"], tpr["macro"])
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


def DET_evaluation(predictions):
    true, scores, n_classes = preproc(predictions)
    fpr = dict()
    fnr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], fnr[i], _ = metrics.det_curve(true[:, i], scores[:, i])
        plt.plot(fpr[i], fnr[i], color='darkorange', lw=2)
        # print('AUC for Class {}: {}'.format(i + 1, metrics.auc(fpr[i], fnr[i])))

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("False Negative Rate")
    plt.title("Some extension of Receiver operating characteristic to multiclass")
    plt.legend(loc="lower right")
    plt.savefig("plot/DET_curve.svg", dpi=1200)


"""
def CMC_evaluation(predictions):
    x = pd.read_csv(predictions)
    labels = list(x.REAL)
    predictions = list(x.PREDICTED)
    ranks = np.zeros(len(labels))
    for i in range(len(labels)):
        if labels[i] in predictions[i]:
            firstOccurance = np.argmax(predictions[i] == labels[i])
            for j in range(firstOccurance, len(labels)):
                ranks[j] += 1

    cmcScores = [float(i) / float(len(labels)) for i in ranks]
    plt.plot(ranks, cmcScores)
    plt.xlabel('Ranks')
    plt.ylabel('cmcScores')
    plt.show()
"""
