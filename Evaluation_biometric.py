import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import train_test_split

from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def eval(dataset="datasets/balanced_dataset.csv"):
    X = pd.read_csv(dataset)

    # encode categorical value
    enc = LabelEncoder()
    enc.classes_ = np.load('classes.npy', allow_pickle=True)
    y = enc.fit_transform(X.pop('PATIENT_NAME'))

    n_classes = len(set(y))

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)

    clf = OneVsRestClassifier(RandomForestClassifier(n_estimators=1400, max_features='auto', max_depth=100,
                                                     min_samples_split=2, min_samples_leaf=1, bootstrap=True))

    clf.fit(X_train, y_train)
    y_score = clf.predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        # errore qua
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()


if __name__ == '__main__':
    eval()
