from SVCModel import SVC_classifier

if __name__ == '__main__':
    dataset = "ptb-diagnostic-ecg-database-1.0.0/"
    data_transformed_file = "data/data_raw.npz"
    new_features_file = "data/dataset_processed.csv"
    feature_extraction_dataset = "data/feature_extracted.csv"
    feature_selection_dataset = "data/feature_selected.csv"
    normalized_dataset = "data/normalized_dataset.csv"
    balanced_dataset = "data/balanced_dataset.csv"

    print("---------- PREPROCESSING THE DATASET")
    # data_preprocessing(dataset, data_transformed_file, new_features_file, feature_extraction_dataset,
    #                     feature_selection_dataset, normalized_dataset, balanced_dataset)

    print("---------- TRAINING THE CLASSIFIER")
    # classifier(balanced_dataset)
    SVC_classifier(balanced_dataset)
