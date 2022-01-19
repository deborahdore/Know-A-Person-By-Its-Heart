import joblib
import numpy as np
import pandas as pd
from sklearn import (
    metrics,
    svm,
    tree,
    ensemble, neural_network,
)
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def work(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    report = pd.DataFrame(metrics.classification_report(y_test, y_pred, zero_division=0, output_dict=True)).T
    return name, score, report


def classifier(dataset, predictions):
    X = pd.read_csv(dataset)

    # encode categorical value
    enc = LabelEncoder()
    y = enc.fit_transform(X.pop('PATIENT_NAME'))

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)

    models = {
        'svm': svm.SVC(),
        'dtree': tree.DecisionTreeClassifier(),
        'mlpn': neural_network.MLPClassifier(),
        'randforest': ensemble.RandomForestClassifier(),
        'adaboost': ensemble.AdaBoostClassifier(),
        'gradient': ensemble.GradientBoostingClassifier()
    }

    res = joblib.Parallel(n_jobs=len(models), verbose=2)(
        joblib.delayed(work)(name, model, X_train, y_train, X_test, y_test) for name, model in models.items()
    )

    best_model = ""
    max_score = -1.0

    for i in range(len(models)):
        print("Model name: ", res[i][0])
        print("Score", res[i][1])
        print("Report")
        print(res[i][2])

        if float(res[i][1]) > max_score:
            best_model = res[i][0]
            max_score = res[i][1]

    print("The best model is:", best_model, "with score", max_score)

    model = models.get(best_model)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    parameters = {'n_estimators': n_estimators,
                  'max_features': max_features,
                  'max_depth': max_depth,
                  'min_samples_split': min_samples_split,
                  'min_samples_leaf': min_samples_leaf,
                  'bootstrap': bootstrap}

    clf = RandomizedSearchCV(estimator=model, param_distributions=parameters, n_iter=500, cv=3, verbose=2,
                             random_state=42, n_jobs=-1)

    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))
    print(clf.best_params_)
    print(clf.score(X_test, y_test))

    joblib.dump(clf, best_model + '.pkl')

    y_pred = clf.predict(X_test)

    y_pred = enc.inverse_transform(y_pred)
    y_test = enc.inverse_transform(y_test)

    new_df = pd.DataFrame(y_test, columns=['REAL'])
    new_df.insert(0, "PREDICTED", y_pred)

    new_df.to_csv(predictions, index=False)

    # {'n_estimators': 200, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'auto', 'max_depth': 50,
    #  'bootstrap': True}