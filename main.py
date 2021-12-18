from DataPreprocessing import data_preprocessing
from LongShortTermMemory import LongShortTermMemory

if __name__ == '__main__':
    dataset = "ptb-diagnostic-ecg-database-1.0.0/"
    data_transformed_file = "data/data_raw.npz"
    new_features_file = "data/dataset_processed.csv"
    distance_dataset_file = "data/distance_dataset.csv"
    normalized_dataset = "data/normalized_dataset.csv"

    print("PREPROCESSING THE DATASET")
    data_preprocessing(dataset, data_transformed_file, new_features_file, normalized_dataset)

    print("FIRST MODEL: DECISION TREE WITH ADABOOST")
    # classifier(normalized_dataset)
