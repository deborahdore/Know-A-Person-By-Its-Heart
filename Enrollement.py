from pathlib import Path

import neurokit2 as nk
from numpy import mean
from scipy.signal import lfilter, find_peaks, peak_widths, peak_prominences

from DataProcessing import k_nearest_neighbour_on_waves, get_time, get_amplitude, get_distance, get_slope, get_angle
from Filters import HighPassFilter, BandStopFilter, LowPassFilter, SmoothSignal


def enrollement(filename):
    enroll_file = open(filename, "r")
    patient_name = ""
    signal_list = []
    for index, line in enumerate(enroll_file):
        if index == 0:
            key, value = line.split(",")
            patient_name = value
        if index > 12:
            # signal acquisition
            signal_list.append(float(line.replace(",", ".")) / 1000)
    denoised_ecg = lfilter(HighPassFilter(), 1, signal_list)
    denoised_ecg = lfilter(BandStopFilter(), 1, denoised_ecg)
    denoised_ecg = lfilter(LowPassFilter(), 1, denoised_ecg)

    cleaned_signal = SmoothSignal(denoised_ecg)

    # fig, ax = plt.subplots()
    # ax.plot(signal[3000:5000], label="Original Signal")
    # ax.plot(cleaned_signal[5000:7000], label="Cleaned Signal")
    # fig.legend()
    # plt.show()

    # only keep best r peaks with prominence = 1
    r_peak, _ = find_peaks(cleaned_signal, prominence=0.25, distance=100)

    # plt.plot(cleaned_signal)
    # plt.plot(r_peak, cleaned_signal[r_peak], "x")
    # plt.xlim(2000, 6000)
    # plt.show()

    # discard signals that have too few r peaks detected
    if len(r_peak) < 15:
        print("patient:", patient_name)
        print("!!!! Enrollment not successful - non enough peaks")
        return

    signal_dwt, waves_dwt = nk.ecg_delineate(cleaned_signal, rpeaks=r_peak, sampling_rate=512, method="dwt",
                                             show=False,
                                             show_type='peaks')

    t_peaks = k_nearest_neighbour_on_waves(waves_dwt['ECG_T_Peaks'])
    p_peaks = k_nearest_neighbour_on_waves(waves_dwt['ECG_P_Peaks'])
    q_peaks = k_nearest_neighbour_on_waves(waves_dwt['ECG_Q_Peaks'])
    s_peaks = k_nearest_neighbour_on_waves(waves_dwt['ECG_S_Peaks'])
    r_peak = k_nearest_neighbour_on_waves(r_peak)

    Tx = mean(peak_widths(cleaned_signal, t_peaks))
    Px = mean(peak_widths(cleaned_signal, p_peaks))
    Qx = mean(peak_widths(cleaned_signal, q_peaks))
    Sx = mean(peak_widths(cleaned_signal, s_peaks))

    Ty = mean(peak_prominences(cleaned_signal, t_peaks))
    Py = mean(peak_prominences(cleaned_signal, p_peaks))
    Qy = mean(peak_prominences(cleaned_signal, q_peaks))
    Sy = mean(peak_prominences(cleaned_signal, s_peaks))

    final_peaks = []
    final_peaks.extend(p_peaks)
    final_peaks.extend(t_peaks)
    final_peaks.extend(q_peaks)
    final_peaks.extend(r_peak)
    final_peaks.extend(s_peaks)
    final_peaks.sort()

    # continue only if all peaks were extracted
    if len(final_peaks) % 5 != 0:
        print("patient:", patient_name)
        print("!!!! Enrollment not successful - incorrect number of peaks")
        return

    features_time = [Tx, Px, Qx, Sx]
    features_time.extend(get_time(final_peaks, cleaned_signal))
    features_amplitude = [Ty, Py, Qy, Sy]
    features_amplitude.extend(get_amplitude(final_peaks, cleaned_signal))
    features_distance = get_distance(final_peaks, cleaned_signal)
    features_slope = get_slope(final_peaks, cleaned_signal)
    features_angle = get_angle(final_peaks, cleaned_signal)

    to_file = []
    to_file.append(patient_name.replace("\n", ""))
    to_file.extend(features_time)
    to_file.extend(features_amplitude)
    to_file.extend(features_distance)
    to_file.extend(features_slope)
    to_file.extend(features_angle)

    # df = pd.read_csv("dataset.csv")

    # patient_datas = pd.Series(to_file, index=df.columns)
    #
    # df = df.append(patient_datas, ignore_index=True)


if __name__ == '__main__':
    for p in Path('./enrollements/').glob('*.csv'):
        enrollement("enrollements/" + p.name)
