import neurokit2 as nk
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Retrieve ECG data from data folder (sampling rate= 1000 Hz)
    ecg_signal = nk.data(dataset="ecg_3000hz")['ECG']

    # Extract R-peaks locations
    _, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=3000)

    # Delineate the ECG signal
    _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000, method="peak")

    # Zooming into the first 3 R-peaks, with focus on T_peaks, P-peaks, Q-peaks and S-peaks
    plot = nk.events_plot([waves_peak['ECG_T_Peaks'][:3],
                           waves_peak['ECG_P_Peaks'][:3],
                           waves_peak['ECG_Q_Peaks'][:3],
                           waves_peak['ECG_S_Peaks'][:3]], ecg_signal[:12500])

    # Delineate the ECG signal and visualizing all peaks of ECG complexes
    _, waves_peak = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000, method="peak", show=True,
                                     show_type='peaks')

    print(waves_peak['ECG_S_Peaks'])