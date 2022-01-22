import joblib
import numpy as np
import pandas as pd
from sklearn import (
    metrics,
    svm,
    tree,
    ensemble, neural_network,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from Evaluation import evaluation


def work(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    report = pd.DataFrame(metrics.classification_report(y_test, y_pred, zero_division=0, output_dict=True)).T
    #  cross val to avoid overfitting
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
    n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return name, score, report, n_scores


def train_classifier(dataset, predictions):
    X = pd.read_csv(dataset)

    # encode categorical value
    enc = LabelEncoder()
    y = enc.fit_transform(X.pop('PATIENT_NAME'))
    np.save('classes.npy', enc.classes_)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)

    models = {
        'svm': svm.SVC(),
        'dtree': tree.DecisionTreeClassifier(),
        'mlpn': neural_network.MLPClassifier(),
        'randforest': ensemble.RandomForestClassifier(),
        'adaboost': ensemble.AdaBoostClassifier(),
    }

    res = joblib.Parallel(n_jobs=len(models), verbose=2)(
        joblib.delayed(work)(name, model, X_train, y_train, X_test, y_test) for name, model in models.items()
    )

    best_model = ""
    max_score = -1.0

    for i in range(len(models)):
        print("Model name: ", res[i][0])
        print("Score", np.mean(res[i][3]))
        print("Report")
        print(res[i][2])

        if np.mean(res[i][3]) > max_score:
            best_model = res[i][0]
            max_score = np.mean(res[i][3])

    print("The best model is:", best_model, "with score", max_score)

    model = models.get(best_model)

    parameters = dict()
    if best_model == 'dtree':
        parameters = {'min_samples_leaf': [1, 2, 4],
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [2, 4, 6, 8, 10, 12]}
    elif best_model == 'randforest' or best_model == 'adaboost':
        parameters = {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                      'max_features': ['auto', 'sqrt'],
                      'max_depth': [int(x) for x in np.linspace(10, 110, num=11)],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4],
                      'bootstrap': [True, False]}
    elif best_model == 'svm':
        parameters = {'C': [0.1, 1, 10, 100, 1000],
                      'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                      'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}

    #  TODO iter num
    clf = RandomizedSearchCV(estimator=model, param_distributions=parameters, n_iter=100, cv=3, verbose=2,
                             random_state=42, n_jobs=-1)

    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))
    print(clf.best_params_)
    print(clf.score(X_test, y_test))

    joblib.dump(clf, 'model.joblib', compress=3)

    y_pred = clf.predict(X_test)
    y_scores = clf.predict_proba(X_test)

    y_pred = enc.inverse_transform(y_pred)
    y_test = enc.inverse_transform(y_test)

    new_df = pd.DataFrame(y_test, columns=['REAL'])
    new_df.insert(0, "PREDICTED", y_pred)
    new_df.insert(2, "SCORES", list(y_scores))

    new_df.to_csv(predictions, index=False)

    evaluation(predictions)
