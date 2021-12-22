import math

import matplotlib.pyplot as plt
import neurokit2 as nk
import wfdb
from numpy import mean
from scipy.signal import lfilter, find_peaks, peak_widths, peak_prominences
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

from Filters import HighPassFilter, BandStopFilter, LowPassFilter, SmoothSignal


# Fill nan values with average
# def fill_nan(l):
#     for i in range(len(l)):
#         if math.isnan(l[i]):
#             if i == 0:
#                 l[i] = l[i + 1] / 2
#             elif i == len(l) - 1:
#                 l[i] = l[i - 1]
#             else:
#                 l[i] = int((l[i + 1] + l[i - 1]) / 2)
#     return l[0:-1]


# Get angle between 3 point
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


# Return time, amplitude, distance, slope, and angle features
def get_features(final_peaks, cleaned_signal):
    PQ_time_list = []
    PT_time_list = []
    QS_time_list = []
    QT_time_list = []
    ST_time_list = []
    PS_time_list = []
    PQ_ampl_list = []
    QR_ampl_list = []
    RS_ampl_list = []
    ST_ampl_list = []
    QS_ampl_list = []
    PS_ampl_list = []
    PT_ampl_list = []
    QT_ampl_list = []
    PQ_dist_list = []
    QR_dist_list = []
    RS_dist_list = []
    ST_dist_list = []
    QS_dist_list = []
    PR_dist_list = []
    PQ_slope_list = []
    QR_slope_list = []
    RS_slope_list = []
    ST_slope_list = []
    QS_slope_list = []
    PT_slope_list = []
    PS_slope_list = []
    QT_slope_list = []
    PR_slope_list = []
    PQR_angle_list = []
    QRS_angle_list = []
    RST_angle_list = []
    RQS_angle_list = []
    RSQ_angle_list = []
    RTS_angle_list = []

    for i in range(0, len(final_peaks), 5):
        P = (final_peaks[i], cleaned_signal[final_peaks[i]])
        Q = (final_peaks[i + 1], cleaned_signal[final_peaks[i + 1]])
        R = (final_peaks[i + 2], cleaned_signal[final_peaks[i + 2]])
        S = (final_peaks[i + 3], cleaned_signal[final_peaks[i + 3]])
        T = (final_peaks[i + 4], cleaned_signal[final_peaks[i + 4]])

        PQ_time_list.append(abs(P[0] - Q[0]))
        PT_time_list.append(abs(P[0] - T[0]))
        QS_time_list.append(abs(Q[0] - S[0]))
        QT_time_list.append(abs(Q[0] - T[0]))
        ST_time_list.append(abs(S[0] - T[0]))
        PS_time_list.append(abs(P[0] - S[0]))

        PQ_ampl_list.append(abs(P[1] - Q[1]))
        QR_ampl_list.append(abs(Q[1] - R[1]))
        RS_ampl_list.append(abs(R[1] - S[1]))
        ST_ampl_list.append(abs(S[1] - T[1]))
        QS_ampl_list.append(abs(Q[1] - S[1]))
        PS_ampl_list.append(abs(P[1] - S[1]))
        PT_ampl_list.append(abs(P[1] - T[1]))
        QT_ampl_list.append(abs(Q[1] - T[1]))

        PQ_dist_list.append(distance.euclidean(P, Q))
        QR_dist_list.append(distance.euclidean(Q, R))
        RS_dist_list.append(distance.euclidean(R, S))
        ST_dist_list.append(distance.euclidean(S, T))
        QS_dist_list.append(distance.euclidean(Q, S))
        PR_dist_list.append(distance.euclidean(P, R))

        PQ_slope_list.append((Q[1] - P[1]) / (Q[0] - P[0]))
        QR_slope_list.append((R[1] - Q[1]) / (R[0] - Q[0]))
        RS_slope_list.append((S[1] - R[1]) / (S[0] - R[0]))
        ST_slope_list.append((T[1] - S[1]) / (T[0] - S[0]))
        QS_slope_list.append((S[1] - Q[1]) / (S[0] - Q[0]))
        PT_slope_list.append((T[1] - P[1]) / (T[0] - P[0]))
        PS_slope_list.append((S[1] - P[1]) / (S[0] - P[0]))
        QT_slope_list.append((T[1] - Q[1]) / (T[0] - Q[0]))
        PR_slope_list.append((R[1] - P[1]) / (R[0] - P[0]))

        PQR_angle_list.append(getAngle(P, Q, R))
        QRS_angle_list.append(getAngle(Q, R, S))
        RST_angle_list.append(getAngle(R, S, T))
        RQS_angle_list.append(getAngle(R, Q, S))
        RSQ_angle_list.append(getAngle(R, S, Q))
        RTS_angle_list.append(getAngle(R, T, S))

    # Time features
    PQ_time = mean(PQ_time_list)
    PT_time = mean(PT_time_list)
    QS_time = mean(QS_time_list)
    QT_time = mean(QT_time_list)
    ST_time = mean(ST_time_list)
    PS_time = mean(PS_time_list)
    PQ_QS_time = PT_time / QS_time
    QT_QS_time = QT_time / QS_time

    # Amplitude features
    PQ_ampl = mean(PQ_ampl_list)
    QR_ampl = mean(QR_ampl_list)
    RS_ampl = mean(RS_ampl_list)
    QS_ampl = mean(QS_ampl_list)
    ST_ampl = mean(ST_ampl_list)
    PS_ampl = mean(PS_ampl_list)
    PT_ampl = mean(PT_ampl_list)
    QT_ampl = mean(QT_ampl_list)
    ST_QS_ampl = ST_ampl / QS_ampl
    RS_QR_ampl = RS_ampl / QR_ampl
    PQ_QS_ampl = PQ_ampl / QS_ampl
    PQ_QT_ampl = PQ_ampl / QT_ampl
    PQ_PS_ampl = PQ_ampl / PS_ampl
    PQ_QR_ampl = PQ_ampl / QR_ampl
    PQ_RS_ampl = PQ_ampl / RS_ampl
    RS_QS_ampl = RS_ampl / QS_ampl
    RS_QT_ampl = RS_ampl / QT_ampl
    ST_PQ_ampl = ST_ampl / PQ_ampl
    ST_QT_ampl = ST_ampl / QT_ampl

    # Distance features
    PQ_dist = mean(PQ_dist_list)
    QR_dist = mean(QR_dist_list)
    RS_dist = mean(RS_dist_list)
    ST_dist = mean(ST_dist_list)
    QS_dist = mean(QS_dist_list)
    PR_dist = mean(PR_dist_list)
    ST_QS_dist = ST_dist / QS_dist
    RS_QR_dist = RS_dist / QR_dist

    # Slope features
    PQ_slope = mean(PQ_slope_list)
    QR_slope = mean(QR_slope_list)
    RS_slope = mean(RS_slope_list)
    ST_slope = mean(ST_slope_list)
    QS_slope = mean(QS_slope_list)
    PT_slope = mean(PT_slope_list)
    PS_slope = mean(PS_slope_list)
    QT_slope = mean(QT_slope_list)
    PR_slope = mean(PR_slope_list)

    # Angle features
    PQR_angle = mean(PQR_angle_list)
    QRS_angle = mean(QRS_angle_list)
    RST_angle = mean(RST_angle_list)
    RQS_angle = mean(RQS_angle_list)
    RSQ_angle = mean(RSQ_angle_list)
    RTS_angle = mean(RTS_angle_list)

    return [PQ_time, PT_time, QS_time, QT_time, ST_time, PS_time, PQ_QS_time,
            QT_QS_time, PQ_ampl, QR_ampl, RS_ampl, QS_ampl, ST_ampl, PS_ampl,
            PT_ampl, QT_ampl, ST_QS_ampl, RS_QR_ampl, PQ_QS_ampl, PQ_QT_ampl,
            PQ_PS_ampl, PQ_QR_ampl, PQ_RS_ampl, RS_QS_ampl, RS_QT_ampl, ST_PQ_ampl,
            ST_QT_ampl, PQ_dist, QR_dist, RS_dist, ST_dist, QS_dist, PR_dist,
            ST_QS_dist, RS_QR_dist, PQ_slope, QR_slope, RS_slope, ST_slope, QS_slope,
            PT_slope, PS_slope, QT_slope, PR_slope, PQR_angle, QRS_angle, RST_angle,
            RQS_angle, RSQ_angle, RTS_angle]


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

    kn = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
    t_peaks = kn.fit(waves_dwt['ECG_T_Peaks'])
    p_peaks = kn.fit(waves_dwt['ECG_P_Peaks'])
    q_peaks = kn.fit(waves_dwt['ECG_Q_Peaks'])
    s_peaks = kn.fit(waves_dwt['ECG_S_Peaks'])
    r_peak = kn.fit(list(r_peak))

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
    plt.plot(r_peak, cleaned_signal[r_peak], "x", label="r_peak")
    plt.plot(q_peaks, cleaned_signal[q_peaks], "x", label="q_peaks")
    plt.plot(s_peaks, cleaned_signal[s_peaks], "x", label="s_peak")

    plt.legend()
    plt.xlim(1500, 5000)
    plt.grid(True)

    # contiene tutti i picchi in ordine P,Q,R,S,T in ripetizione
    final_peaks = []
    final_peaks.extend(p_peaks)
    final_peaks.extend(t_peaks)
    final_peaks.extend(q_peaks)
    final_peaks.extend(r_peak)
    final_peaks.extend(s_peaks)
    final_peaks.sort()

    features = get_features(final_peaks, cleaned_signal)
    print(features)

    plt.show()
