import neurokit2 as nk
import csv
import os.path
import glob
import numpy as np
import wfdb
from neurokit2 import NeuroKitWarning


def extract_peak(ecg_signal_file):
    data = np.load(ecg_signal_file)
    matr_data = []
    for patient_name in data.files:
        try:
            _, rpeak = nk.ecg_peaks(data[patient_name], sampling_rate=3000)
            # Delineate the ECG signal and visualizing all peaks of ECG complexes
            _, waves_peak = nk.ecg_delineate(data[patient_name], rpeak, sampling_rate=3000, method="dwt", show=True,
                                             show_type='peaks')

            r_peak = rpeak['ECG_R_Peaks']
            p_peak = waves_peak['ECG_P_Peaks']
            t_peak = waves_peak['ECG_T_Peaks']
            q_peak = waves_peak['ECG_Q_Peaks']
            s_peak = waves_peak['ECG_S_Peaks']

            for index in range(len(r_peak)):
                matr_data.append(
                    [patient_name, r_peak[index], p_peak[index], t_peak[index], q_peak[index], s_peak[index]])
        except NeuroKitWarning:
            print("Too few peaks detected for patient:", patient_name)
        except RuntimeWarning:
            print("Warning extracting peaks from patient:", patient_name)
        except RuntimeError:
            print("Errors extracting peaks from patient:", patient_name)
    return matr_data


def write_to_file(matr_data, file):
    exists = os.path.exists(file)
    header = ['PATIENT_NAME', 'R_PEAK', 'P_PEAK', 'T_PEAK', 'Q_PEAK', 'S_PEAK']
    with open(file, "a") as ecg_samples:
        writer = csv.writer(ecg_samples)
        if not exists:
            writer.writerow(header)
        print(matr_data)
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


if __name__ == '__main__':
    dataset = "ptb-diagnostic-ecg-database-1.0.0/"
    data_transformed_file = "data_raw.npz"
    new_features_file = "dataset_processed.csv"
    transform_ecg_data(dataset, data_transformed_file)
    write_to_file(extract_peak(data_transformed_file), new_features_file)
