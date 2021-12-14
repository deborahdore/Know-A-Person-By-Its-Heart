import pandas as pd
from sklearn.model_selection import train_test_split


def AdaBoost(dataset):
    df = pd.read_csv(dataset)

    y = df.pop('PATIENT_NAME')
    x = df

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    