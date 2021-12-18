import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_val_score, train_test_split


def classifier(dataset):
    print("Accuracy evaluation using default hyperparameters")

    df = pd.read_csv(dataset).sample(frac=0.10)
    y = df.pop('PATIENT_NAME')
    x = df

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    # default hyperparameters
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=0.05)
    # evaluate the model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(classifier, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    # report performance
    print('Accuracy: %.3f (%.3f) with default hyperparameters' % (np.mean(n_scores), np.std(n_scores)))

    print("Now using GridSearch")
    grid = dict()
    grid['n_estimators'] = [10, 50, 100, 500]
    grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]

    # evaluation
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid_search = GridSearchCV(estimator=classifier, param_grid=grid, cv=cv, scoring='accuracy')

    result = grid_search.fit(x_train, y_train)

    print("Best: %f using %s" % (result.best_score_, result.best_params_))

    # summarize all scores that were evaluated
    means = result.cv_results_['mean_test_score']
    stds = result.cv_results_['std_test_score']
    params = result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
