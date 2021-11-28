import neurokit2 as nk
import csv
import os.path
import glob
import pandas as pd
import wfdb
import numpy as np
from sklearn.impute import KNNImputer


def extract_peak(ecg_signal_file):
    data = np.load(ecg_signal_file)
    matr_data = []
    for patient_name in data.files:
        try:
            _, rpeak = nk.ecg_peaks(data[patient_name], sampling_rate=3000)
            # Delineate the ECG signal and visualizing all peaks of ECG complexes
            _, waves_peak = nk.ecg_delineate(data[patient_name], rpeak, sampling_rate=3000, method="dwt", show=False,
                                             show_type='peaks')

            r_peak = rpeak['ECG_R_Peaks']
            p_peak = waves_peak['ECG_P_Peaks']
            t_peak = waves_peak['ECG_T_Peaks']
            q_peak = waves_peak['ECG_Q_Peaks']
            s_peak = waves_peak['ECG_S_Peaks']

            total_len = min(len(r_peak), len(p_peak), len(t_peak), len(q_peak), len(s_peak))
            for index in range(total_len):
                matr_data.append(
                    [patient_name, r_peak[index], p_peak[index], t_peak[index], q_peak[index], s_peak[index]])
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
    return matr_data


def write_to_file(matr_data, file):
    exists = os.path.exists(file)
    header = ['PATIENT_NAME', 'R_PEAK', 'P_PEAK', 'T_PEAK', 'Q_PEAK', 'S_PEAK']
    with open(file, "a") as ecg_samples:
        writer = csv.writer(ecg_samples)
        if not exists:
            writer.writerow(header)
        for row in matr_data:
            writer.writerow(row)


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


def check_dataset(data):
    # compute k-nearest neighbour
    df = pd.read_csv(data)
    read_data = df[['R_PEAK', 'P_PEAK', 'T_PEAK', 'Q_PEAK', 'S_PEAK']]
    imputer = KNNImputer(n_neighbors=2, weights='uniform')
    read_data = pd.DataFrame(imputer.fit_transform(read_data),
                             columns=['R_PEAK', 'P_PEAK', 'T_PEAK', 'Q_PEAK', 'S_PEAK'])
    new_df = df[['PATIENT_NAME']]
    new_df = new_df.join(read_data)
    new_df.to_csv(data, index=False)


if __name__ == '__main__':
    dataset = "ptb-diagnostic-ecg-database-1.0.0/"
    data_transformed_file = "data_raw.npz"
    new_features_file = "dataset_processed.csv"
    transform_ecg_data(dataset, data_transformed_file)
    write_to_file(extract_peak(data_transformed_file), new_features_file)
    check_dataset(new_features_file)
