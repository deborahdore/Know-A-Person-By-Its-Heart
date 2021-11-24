import neurokit2 as nk
import numpy as np
import csv
import os.path
from scipy.stats import zscore


# extract the peak, for now the database is of example
def extract_peak(ecg_signal=nk.data(dataset="ecg_3000hz")['ECG']):
    _, rpeak = nk.ecg_peaks(ecg_signal, sampling_rate=3000)
    # Delineate the ECG signal and visualizing all peaks of ECG complexes
    _, waves_peak = nk.ecg_delineate(ecg_signal, rpeak, sampling_rate=3000, method="dwt", show=True,
                                     show_type='peaks')

    # normalization
    r_normalized = zscore(rpeak['ECG_R_Peaks'], nan_policy='omit')
    p_normalized = zscore(waves_peak['ECG_P_Peaks'], nan_policy='omit')
    t_normalized = zscore(waves_peak['ECG_T_Peaks'], nan_policy='omit')
    q_normalized = zscore(waves_peak['ECG_Q_Peaks'], nan_policy='omit')
    s_normalized = zscore(waves_peak['ECG_S_Peaks'], nan_policy='omit')

    # extract the mean
    t_peak = np.nanmean(t_normalized)
    q_peak = np.nanmean(q_normalized)
    s_peak = np.nanmean(s_normalized)
    p_peak = np.nanmean(p_normalized)
    r_peak = np.nanmean(r_normalized)

    peaks = np.array([q_peak - r_peak,
                      r_peak - s_peak,
                      p_peak - r_peak,
                      r_peak - t_peak,
                      s_peak - t_peak,
                      p_peak - q_peak,
                      p_peak - t_peak,
                      ])

    return peaks


def write_to_file(filename, dictionary, header):
    exists = os.path.exists(filename)
    with open(filename, "a") as ecg_samples:
        writer = csv.writer(ecg_samples)
        if not exists:
            writer.writerow(header)
        writer.writerow(dictionary)


if __name__ == '__main__':
    headers = ['RQ', 'RS', 'RP', 'RT', 'ST', 'PQ', 'PT']
    write_to_file("ecg_samples.csv", extract_peak(), headers)
