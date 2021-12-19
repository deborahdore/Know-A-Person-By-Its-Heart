import csv
import glob
from collections import Counter

import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb
from imblearn.over_sampling import RandomOverSampler
from matplotlib import pyplot
from scipy.stats import zscore
from sklearn.impute import KNNImputer
from sklearn.inspection import permutation_importance
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder
from tsfresh import extract_relevant_features


def transform_ecg_data(p, new_file_name):
    template = p + "*/*.hea"
    file_list = glob.glob(template)

    data_raw = {}

    for file in file_list:
        patient, record_id = file[len(p):-4].split("/")
        m = wfdb.rdsamp(file[:-4], channel_names=["v6"])[0]
        arr = np.array([m[i][0] for i in range(len(m))])
        key = patient + "/" + record_id
        data_raw[key] = arr.astype(np.float32)

    np.savez_compressed(new_file_name, **data_raw)


def ecg_processing(ecg_signal_file, new_dataset_file):
    data = np.load(ecg_signal_file)
    matr_data = []
    for patient_name in data.files:
        try:

            signals, info = nk.ecg_process(data[patient_name], sampling_rate=1000)
            r_peak = info["ECG_R_Peaks"]
            cleaned_ecg = signals["ECG_Clean"]

            # Delineate the ECG signal and visualizing all peaks of ECG complexes
            signal_cwt, waves_peak = nk.ecg_delineate(cleaned_ecg, r_peak, sampling_rate=1000, method="cwt", show=False,
                                                      show_type='all')
            # plt.show()

            p_peak = waves_peak['ECG_P_Peaks']
            t_peak = waves_peak['ECG_T_Peaks']
            q_peak = waves_peak['ECG_Q_Peaks']
            s_peak = waves_peak['ECG_S_Peaks']

            p_onset_peak = waves_peak['ECG_P_Onsets']
            p_offset_peak = waves_peak['ECG_P_Offsets']
            r_onset_peak = waves_peak['ECG_R_Onsets']
            r_offset_peak = waves_peak['ECG_R_Offsets']
            t_onset_peak = waves_peak['ECG_T_Onsets']
            t_offset_peak = waves_peak['ECG_T_Offsets']

            matr_data.append(
                [patient_name.split('/')[0], np.mean(p_onset_peak), np.mean(p_peak), np.mean(p_offset_peak),
                 np.mean(q_peak),
                 np.mean(r_onset_peak), np.mean(r_offset_peak), np.mean(s_peak), np.mean(t_onset_peak),
                 np.mean(t_peak), np.mean(t_offset_peak)])

        except ValueError as error:
            print("Errors extracting peaks from patient:", patient_name)
            print(error)
        except ZeroDivisionError as error:
            print("Errors extracting peaks from patient:", patient_name)
            print(error)
        except RuntimeError as error:
            print("Errors extracting peaks from patient:", patient_name)
            print(error)
        except IndexError as error:
            print("Errors extracting peaks from patient:", patient_name)
            print(error)

    print("Create new dataset")
    header = ['PATIENT_NAME', 'ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets', 'ECG_Q_Peaks', 'ECG_R_Onsets',
              'ECG_R_Offsets', 'ECG_S_Peaks', 'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets']

    with open(new_dataset_file, "w") as ecg_samples:
        writer = csv.writer(ecg_samples)
        writer.writerow(header)
        writer.writerows(matr_data)


def compute_k_nearest_neighbour(data):
    # compute k-nearest neighbour
    df = pd.read_csv(data)
    read_data = df[
        ['ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets', 'ECG_Q_Peaks', 'ECG_R_Onsets', 'ECG_R_Offsets', 'ECG_S_Peaks',
         'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets']]
    imputer = KNNImputer(n_neighbors=2, weights='uniform')
    read_data = pd.DataFrame(imputer.fit_transform(read_data),
                             columns=['ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets', 'ECG_Q_Peaks', 'ECG_R_Onsets',
                                      'ECG_R_Offsets', 'ECG_S_Peaks', 'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets'])
    new_df = df[['PATIENT_NAME']]
    new_df = new_df.join(read_data)
    new_df.to_csv(data, index=True, index_label="id")


def feature_extraction(dataset, feature_extraction_csv):
    X = pd.read_csv(dataset)
    y = X['PATIENT_NAME']
    features_filtered_direct = extract_relevant_features(X, y, column_id='id', column_sort='PATIENT_NAME')
    df = pd.concat([pd.DataFrame(y), features_filtered_direct], axis=1)
    df.to_csv(feature_extraction_csv, index=False)


def feature_importance(dataset, feature_selection_dataset):
    # the permutation feature importance measures the increase in the prediction error of the model after we
    # permutated the features'values
    X = pd.read_csv(dataset)
    y = X.pop('PATIENT_NAME')
    y_encoded = LabelEncoder().fit_transform(y)
    model = KNeighborsRegressor()
    model.fit(X, y_encoded)
    # perform permutation importance
    results = permutation_importance(model, X, y_encoded, scoring='neg_mean_squared_error')
    importance = results.importances_mean
    suggested_features = []
    for i, v in enumerate(importance):
        if v > 1:
            suggested_features.append(i)
            print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    pyplot.bar([x > 1 for x in range(len(importance))], importance)
    pyplot.show()
    print("Suggested features: ", X.columns[suggested_features])

    new_df = pd.concat([pd.DataFrame(y, columns=['PATIENT_NAME']), X[X.columns[[n for n in suggested_features]]]],
                       axis=1)
    new_df.to_csv(feature_selection_dataset, index=False)


def ecg_normalization(dataset, normalized_dataset):
    #  tanh normalization
    df = pd.read_csv(dataset)
    header = df.columns
    label = df.pop('PATIENT_NAME')
    unnormalizedData = df.to_numpy()

    m = np.mean(unnormalizedData, axis=0)
    std = np.std(unnormalizedData, axis=0)

    data = 0.5 * (np.tanh(0.01 * ((unnormalizedData - m) / std)) + 1)

    normalized_df = pd.concat([label, pd.DataFrame(data)], axis=1)
    normalized_df.to_csv(normalized_dataset, index=False, header=header)


def remove_outliers(dataset):
    # remove outliers
    df = pd.read_csv(dataset)
    df_label = df.pop('PATIENT_NAME')
    z_scores = zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df_removed_outliers = pd.concat([df_label, df[filtered_entries]], axis=1)
    df_removed_outliers.to_csv(dataset, index=False)


def balance_dataset(dataset, balanced_dataset):
    X = pd.read_csv(dataset)
    y = X.pop('PATIENT_NAME')
    counter = Counter(y)
    print("Imbalanced Classes:", counter)
    over = RandomOverSampler()
    X_over, y_over = over.fit_resample(X, y)
    counter = Counter(y_over)
    print("Balanced Classes:", counter)
    df = pd.concat([y_over, X_over], axis=1)
    df.to_csv(balanced_dataset, index=False)


def data_preprocessing(dataset, data_transformed_file, new_features_file, feature_extraction_dataset,
                       feature_selection_dataset, normalized_dataset, balanced_dataset):
    print("---------- Extract ECG")
    transform_ecg_data(dataset, data_transformed_file)
    print("---------- Processing the ECG.. filtering, denoising and clearing the signals")
    ecg_processing(data_transformed_file, new_features_file)
    print("---------- Removing nan values from the dataset using K-Nearest-Neighbour")
    compute_k_nearest_neighbour(new_features_file)
    print("---------- Extracting relevant features from the dataset")
    feature_extraction(new_features_file, feature_extraction_dataset)
    print("---------- Selecting features with high importance between the ones extracted")
    feature_importance(feature_extraction_dataset, feature_selection_dataset)
    print("---------- Features normalization using tahn normalization")
    ecg_normalization(feature_selection_dataset, normalized_dataset)
    print("---------- RANDOM OVER SAMPLER for balancing dataset")
    balance_dataset(normalized_dataset, balanced_dataset)
