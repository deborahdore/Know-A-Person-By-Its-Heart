import sys

from Classifier import train_classifier
from DataProcessing import data_processing
from Enrollement import start_enrollment
from Predictions import predict_class

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
        train_classifier(balanced_dataset, predictions)
    if argument == 'enroll':
        # enroll
        start_enrollment(dataset, balanced_dataset, analyzed_dataset, predictions)
    if argument == 'predict':
        #  predict
        predict_class()
