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
    # cleaned_signal = cleaned_signal[2400:]

    plt.clf()
    # plt.plot(signal, label="RAW ECG")
    # plt.plot(cleaned_signal, label="Cleaned ECG", color='k')
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # only keep best r peaks with prominence = 1
    r_peak, _ = find_peaks(cleaned_signal, prominence=1, distance=150)

    # segment the signal
    signal_dwt, waves_cwt = nk.ecg_delineate(cleaned_signal, rpeaks=r_peak, sampling_rate=1000, method="cwt",
                                             show=False,
                                             show_type='peaks')

    t_peaks = [n for n in waves_cwt['ECG_T_Peaks'] if not math.isnan(n)]
    p_peaks = [n for n in waves_cwt['ECG_P_Peaks'] if not math.isnan(n)]
    q_peak = [n for n in waves_cwt['ECG_Q_Peaks'] if not math.isnan(n)]
    s_peak = [n for n in waves_cwt['ECG_S_Peaks'] if not math.isnan(n)]

    #
    # plot = nk.events_plot([waves_cwt['ECG_T_Peaks'],
    #                        waves_cwt['ECG_P_Peaks'],
    #                        waves_cwt['ECG_Q_Peaks'],
    #                        waves_cwt['ECG_S_Peaks']], cleaned_signal[:12500])
    # plt.xlim(11000, 12500)
    # plt.ylim(-0.1, 0.2)

    Tx = mean(peak_widths(cleaned_signal, t_peaks))
    Px = mean(peak_widths(cleaned_signal, p_peaks))
    Qx = mean(peak_widths(cleaned_signal, q_peak))
    Sx = mean(peak_widths(cleaned_signal, s_peak))

    Ty = mean(peak_prominences(cleaned_signal, t_peaks))
    Py = mean(peak_prominences(cleaned_signal, p_peaks))
    Qy = mean(peak_prominences(cleaned_signal, q_peak))
    Sy = mean(peak_prominences(cleaned_signal, s_peak))

    # plt.plot(cleaned_signal)
    # plt.plot(t_peaks, cleaned_signal[t_peaks], "x", label="t_peaks")
    # plt.plot(p_peaks, cleaned_signal[p_peaks], "x", label="p_peaks")
    # plt.plot(q_peak, cleaned_signal[q_peak], "x", label="q_peak")
    # plt.plot(s_peak, cleaned_signal[s_peak], "x", label="s_peak")
    #
    # plt.legend()
    # plt.grid(True)
    # plt.xlim(0,5000)
    # plt.ylim(-1,1.5)

    plt.show()
