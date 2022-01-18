import joblib
import pandas as pd
from sklearn import (
    model_selection,
    metrics,
    ensemble,
)
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

X_train = []
y_train = []
X_test = []
y_test = []


def work(name, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    report = metrics.classification_report(y_test, y_pred, zero_division=0)
    cm = metrics.confusion_matrix(y_test, y_pred)

    kf = KFold(n_splits=3)
    cv = model_selection.cross_val_score(model, X_test, y_test, n_jobs=2, cv=kf)

    return name, score, report, cm, cv


def main(dataset):
    X = pd.read_csv(dataset)

    # encode categorical value
    enc = LabelEncoder()
    y = enc.fit_transform(X.pop('PATIENT_NAME'))

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    global X_train
    global X_test
    global y_train
    global y_test

    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=42)

    models = {
        # 'svm': svm.SVC(),
        # 'dtree': tree.DecisionTreeClassifier(),
        # 'randforest': ensemble.RandomForestClassifier(),
        # 'mlpn': neural_network.MLPClassifier(),
        # 'adaboost': ensemble.AdaBoostClassifier(),
        'gradient': ensemble.GradientBoostingClassifier()
    }

    res = joblib.Parallel(n_jobs=len(models), verbose=2)(
        joblib.delayed(work)(name, model) for name, model in models.items()
    )
    print(res)
    #
    # parameters = {
    #     "loss": ["deviance"],
    #     "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    #     "min_samples_split": np.linspace(0.1, 0.5, 12),
    #     "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    #     "max_depth": [3, 5, 8],
    #     "max_features": ["log2", "sqrt"],
    #     "criterion": ["friedman_mse", "mae"],
    #     "subsample": [0.5, 0.618, 0.8, 0.85, 0.9, 0.95, 1.0],
    #     "n_estimators": [10]
    # }
    #
    # clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=5, n_jobs=-1)
    #
    # clf.fit(X_train, y_train)
    # print(clf.score(X_train, y_train))
    # print(clf.best_params_)
    # print(clf.score(X_test, y_test))


if __name__ == '__main__':
    main("balanced_dataset.csv")
