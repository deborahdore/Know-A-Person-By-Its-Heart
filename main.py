from DataProcessing import data_processing
from Enrollement import start_enrollment

if __name__ == '__main__':
    base_path = "ptb-diagnostic-ecg-database-1.0.0/"
    dataset = "datasets/dataset.csv"
    balanced_dataset = "datasets/balanced_dataset.csv"
    analyzed_dataset = "datasets/analyzed_dataset.csv"
    predictions = "datasets/predictions_dataset.csv"
    data_processing(base_path, dataset, balanced_dataset, analyzed_dataset)

    # classifier(analyzed_dataset, predictions)
    start_enrollment(dataset, balanced_dataset, analyzed_dataset, predictions)
