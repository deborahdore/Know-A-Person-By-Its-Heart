import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

def AdaBoost(dataset):
    df = pd.read_csv(dataset)

    y = df.pop('PATIENT_NAME')
    x = df

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    
    abc = AdaBoostClassifier(n_estimators=50, learning_rate=0.05)
    model = abc.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


def GridSearch(dataset):
    df = pd.read_csv(dataset)

    y = df.pop('PATIENT_NAME')
    x = df

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    
    abc = AdaBoostClassifier(base_estimator=DecisionTreeClassifier())

    parameters = {'base_estimator__max_depth':[i for i in range(2,11,2)],
                  'base_estimator__min_samples_leaf':[5,10],
                  'n_estimators':[10,50,250,1000],
                  'learning_rate':[0.01,0.1]}

    clf = GridSearchCV(abc, parameters,verbose=3,scoring='f1',n_jobs=-1)
    clf.fit(x_train, y_train)
    return clf.best_params_