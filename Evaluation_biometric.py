import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MinMaxScaler, label_binarize


def eval(dataset="datasets/balanced_dataset.csv"):
    X = pd.read_csv(dataset)

    # encode categorical value
    classes = X.pop('PATIENT_NAME')
    y = label_binarize(classes, classes=np.unique(classes))

    n_classes = len(np.unique(classes))

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)

    clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=1400, max_features='auto', max_depth=100,
                                                     min_samples_split=2, min_samples_leaf=1, bootstrap=True))
    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)

    ROC_curve(n_classes, y_score, y_test)


def ROC_curve(n_classes, y_score, y_test):
    far = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    frr = dict()
    eer = dict()

    for i in range(n_classes):
        far[i], tpr[i], thresholds[i] = roc_curve(y_test[:, i], y_score[:, i])
        frr[i] = 1 - tpr[i]
        # eer[i] = far[i][np.nanargmin(np.absolute((frr[i] - far[i])))]
        roc_auc[i] = auc(far[i], tpr[i])
    # Plot of a ROC curve for a specific class
    specific_class = [0, 5, 22]
    for i in specific_class:
        plt.figure()
        plt.plot(far[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('plot/ROC/ROC_curve' + str(i) + '.svg', dpi=1200)
        plt.clf()
    # plot overall curve
    plt.figure(figsize=(10, 10))
    for i in range(n_classes):
        plt.plot(far[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.tight_layout()
    plt.savefig('plot/ROC/ROC_curve.svg', dpi=1200)
    plt.clf()

    # plot FAR
    for i in range(n_classes):
        plt.plot(thresholds[i], far[i])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold')
    plt.ylabel('Equal Error Rate')
    plt.title('False Acceptance Rate')
    plt.tight_layout()
    plt.savefig('plot/FAR_FRR/FAR_curve.svg', dpi=1200)
    plt.clf()

    # plot FRR against threshold
    for i in range(n_classes):
        plt.plot(thresholds[i], frr[i])

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Threshold')
    plt.ylabel('False Rejection Rate')
    plt.title('FRR')
    plt.tight_layout()
    plt.savefig('plot/FAR_FRR/FFR_curve.svg', dpi=1200)
    plt.clf()



if __name__ == '__main__':
    eval()
