import csv
import glob
import os.path

import neurokit2 as nk
import numpy as np
import pandas as pd
import wfdb
from scipy.stats import zscore
from sklearn import decomposition
from sklearn.impute import KNNImputer


def transform_ecg_data(p, new_file_name):
    print("Transform ecg dataset")

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
    print("Processing the dataset..")
    data = np.load(ecg_signal_file)
    matr_data = []
    print("Signal cleaning and peak extraction")
    print("Signal cleaning and denoising and peak extraction")
    for patient_name in data.files:
        try:

            signals, info = nk.ecg_process(data[patient_name], sampling_rate=1000)
            r_peak = info["ECG_R_Peaks"]
            cleaned_ecg = signals["ECG_Clean"]

            # Delineate the ECG signal and visualizing all peaks of ECG complexes
            _, waves_peak = nk.ecg_delineate(cleaned_ecg, r_peak, sampling_rate=1000, method="cwt", show=False,
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

            total_len = min(len(r_peak), len(p_peak), len(t_peak), len(q_peak), len(s_peak), len(p_onset_peak),
                            len(p_offset_peak), len(r_onset_peak), len(r_offset_peak), len(t_onset_peak),
                            len(t_offset_peak))
            for index in range(total_len):
                matr_data.append([patient_name, p_onset_peak[index], p_peak[index], p_offset_peak[index], q_peak[index],
                                  r_onset_peak[index], r_offset_peak[index], s_peak[index], t_onset_peak[index],
                                  t_peak[index], t_offset_peak[index]])
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

    exists = os.path.exists(new_dataset_file)
    header = ['PATIENT_NAME', 'ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets', 'ECG_Q_Peaks', 'ECG_R_Onsets',
              'ECG_R_Offsets', 'ECG_S_Peaks', 'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets']
    with open(new_dataset_file, "a") as ecg_samples:
        writer = csv.writer(ecg_samples)
        if not exists:
            writer.writerow(header)
        for row in matr_data:
            writer.writerow(row)


def compute_k_nearest_neighbour(data):
    print("Computing K-Nearest Neighbour to remove nan values")

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
    new_df.to_csv(data, index=False)


def ecg_normalization(dataset, normalized_dataset):
    print("Normalize Dataset using tahn normalization")
    #  tanh normalization
    df = pd.read_csv(dataset)
    label = df.pop('PATIENT_NAME')
    unnormalizedData = df.to_numpy()

    m = np.mean(unnormalizedData, axis=0)
    std = np.std(unnormalizedData, axis=0)

    data = 0.5 * (np.tanh(0.01 * ((unnormalizedData - m) / std)) + 1)

    normalized_df = pd.concat([label, pd.DataFrame(data)], axis=1)
    header = ['PATIENT_NAME', 'ECG_P_Onsets', 'ECG_P_Peaks', 'ECG_P_Offsets', 'ECG_Q_Peaks', 'ECG_R_Onsets',
              'ECG_R_Offsets', 'ECG_S_Peaks', 'ECG_T_Onsets', 'ECG_T_Peaks', 'ECG_T_Offsets']
    normalized_df.to_csv(normalized_dataset, index=False, header=header)


def ecg_feature_selection(dataset, new_dataset):
    print("Feature selection using PCA methodology")
    df = pd.read_csv(dataset)
    y = df.pop('PATIENT_NAME')
    X = df
    svd = decomposition.TruncatedSVD()
    X_pca = svd.fit_transform(X)
    new_df = pd.concat([y, pd.DataFrame(X_pca)], axis=1)
    new_df.to_csv(new_dataset, index=False)

    print("how much information has each feature -> ", svd.explained_variance_ratio_)


def remove_outliers(dataset):
    # remove outliers
    df = pd.read_csv(dataset)
    df_label = df.pop('PATIENT_NAME')
    z_scores = zscore(df)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    df_removed_outliers = pd.concat([df_label, df[filtered_entries]], axis=1)
    df_removed_outliers.to_csv(dataset, index=False)


def format_correctly(energy_file):
    transformed = []
    with open(energy_file, "r") as file:
        reader = csv.reader(file)
        for index, row in enumerate(reader):
            if index == 0:
                continue
            transformed.append([row[0], format(float(row[1]), '.10f'), format(float(row[2]), '.10f')])

    with open(energy_file, "w") as file:
        writer = csv.writer(file)
        writer.writerow(['PATIENT_NAME', 'R1', 'R2'])
        writer.writerows(transformed)


def data_preprocessing(dataset, data_transformed_file, new_features_file, normalized_dataset, feature_reduction_file):
    transform_ecg_data(dataset, data_transformed_file)
    ecg_processing(data_transformed_file, new_features_file)
    # remove nan
    compute_k_nearest_neighbour(new_features_file)
    # normalization
    ecg_normalization(new_features_file, normalized_dataset)
    # feature selection
    ecg_feature_selection(normalized_dataset, feature_reduction_file)
    remove_outliers(feature_reduction_file)
    format_correctly(feature_reduction_file)
