import math

import matplotlib.pyplot as plt
import neurokit2 as nk
import wfdb
from numpy import mean
from scipy.signal import lfilter, find_peaks, peak_widths, peak_prominences

from Filters import HighPassFilter, BandStopFilter, LowPassFilter, SmoothSignal


def main():
    path = "physionet.org/files/ptbdb/1.0.0/patient001/s0010_re"
    record = wfdb.rdrecord(path, channel_names=['v4'])

    # wfdb.plot_wfdb(record=record, time_units='seconds', figsize=(50, 10), ecg_grids='all')
    # plt.show()

    signal = record.p_signal.ravel()
    denoised_ecg = lfilter(HighPassFilter(), 1, signal)
    denoised_ecg = lfilter(BandStopFilter(), 1, denoised_ecg)
    denoised_ecg = lfilter(LowPassFilter(), 1, denoised_ecg)

    cleaned_signal = SmoothSignal(denoised_ecg)

    plt.clf()

    # plt.plot(signal, label="RAW ECG")
    # plt.plot(cleaned_signal, label="Cleaned ECG", color='k')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # only keep best r peaks with prominence = 1
    r_peak, _ = find_peaks(cleaned_signal, prominence=1, distance=150)
    signal_dwt, waves_dwt = nk.ecg_delineate(cleaned_signal, rpeaks=r_peak, sampling_rate=1000, method="dwt", show=True,
                                             show_type='peaks')

    t_peaks = [n for n in waves_dwt['ECG_T_Peaks'] if not math.isnan(n)]
    p_peaks = [n for n in waves_dwt['ECG_P_Peaks'] if not math.isnan(n)]
    q_peaks = [n for n in waves_dwt['ECG_Q_Peaks'] if not math.isnan(n)]
    s_peaks = [n for n in waves_dwt['ECG_S_Peaks'] if not math.isnan(n)]

    Tx = mean(peak_widths(cleaned_signal, t_peaks))
    Px = mean(peak_widths(cleaned_signal, p_peaks))
    Qx = mean(peak_widths(cleaned_signal, q_peaks))
    Sx = mean(peak_widths(cleaned_signal, s_peaks))

    Ty = mean(peak_prominences(cleaned_signal, t_peaks))
    Py = mean(peak_prominences(cleaned_signal, p_peaks))
    Qy = mean(peak_prominences(cleaned_signal, q_peaks))
    Sy = mean(peak_prominences(cleaned_signal, s_peaks))

    plt.clf()
    plt.plot(cleaned_signal)
    plt.plot(t_peaks, cleaned_signal[t_peaks], "x", label="t_peaks")
    plt.plot(p_peaks, cleaned_signal[p_peaks], "x", label="p_peaks")
    plt.plot(q_peaks, cleaned_signal[q_peaks], "x", label="q_peaks")
    plt.plot(s_peaks, cleaned_signal[s_peaks], "x", label="s_peak")
    plt.legend()
    plt.xlim(0, 5000)
    plt.grid(True)

    # contiene tutti i picchi in ordine P,Q,S,T in ripetizione
    final_peaks = p_peaks
    final_peaks.extend(t_peaks)
    final_peaks.extend(q_peaks)
    final_peaks.extend(s_peaks)
    final_peaks.sort()

    plt.show()
