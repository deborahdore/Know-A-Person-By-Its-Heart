import sys

from Classifier import train_classifier
from DataProcessing import data_processing
from Enrollement import start_enrollment, predict_class

if __name__ == '__main__':
    base_path = "ptb-diagnostic-ecg-database-1.0.0/"
    dataset = "datasets/dataset.csv"
    balanced_dataset = "datasets/balanced_dataset.csv"
    analyzed_dataset = "datasets/analyzed_dataset.csv"
    predictions = "datasets/predictions_dataset.csv"

    if len(sys.argv[0:]) != 2:
        print("NOT ENOUGH ARGUMENTS")
        sys.exit()

    argument = sys.argv[1]

    if argument == 'process':
        # process dataset
        data_processing(base_path, dataset, balanced_dataset, analyzed_dataset)
        # train classifier
        train_classifier(analyzed_dataset, predictions)
    if argument == 'enroll':
        # enroll
        start_enrollment(dataset, balanced_dataset, analyzed_dataset, predictions)
    if argument == 'predict':
        #  predict
        predict_class()

    # {'n_estimators': 1400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 80, 'bootstrap': True}
    # 0.8407643312101911