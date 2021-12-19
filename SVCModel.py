import pandas as pd
from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import train_test_split


def evaluate_kernel(X_train, y_train, X_test, y_test, kernel):
    clf = svm.SVC(kernel=kernel)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy", kernel, "Kernel :", metrics.accuracy_score(y_test, y_pred))


def SVC_classifier(dataset):
    print("Accuracy evaluation using default hyperparameters")

    df = pd.read_csv(dataset).sample(frac=0.10)
    y = df.pop('PATIENT_NAME')
    x = df

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    evaluate_kernel(X_train, y_train, X_test, y_test, "linear")
    evaluate_kernel(X_train, y_train, X_test, y_test, "poly")
    evaluate_kernel(X_train, y_train, X_test, y_test, "rbf")
    evaluate_kernel(X_train, y_train, X_test, y_test, "sigmoid")
    evaluate_kernel(X_train, y_train, X_test, y_test, "precomputed")
