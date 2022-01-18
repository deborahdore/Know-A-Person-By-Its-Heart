from DataProcessing import main

if __name__ == '__main__':
    base_path = "ptb-diagnostic-ecg-database-1.0.0/"
    dataset = "dataset.csv"
    balanced_dataset = "balanced_dataset.csv"
    main(base_path, dataset, balanced_dataset)
