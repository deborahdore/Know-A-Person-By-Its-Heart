from Classifier import classifier

if __name__ == '__main__':
    base_path = "ptb-diagnostic-ecg-database-1.0.0/"
    dataset = "dataset.csv"
    balanced_dataset = "balanced_dataset.csv"
    analyzed_dataset = "analyzed_dataset.csv"
    predictions = "predictions_dataset.csv"
    # data_processing(base_path, dataset, balanced_dataset, analyzed_dataset)
    classifier(analyzed_dataset, predictions)
