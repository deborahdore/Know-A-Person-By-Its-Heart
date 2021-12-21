import numpy as np
from scipy.signal import kaiserord, firwin
from scipy.signal import savgol_filter


def BandStopFilter():
    #  minima frequenza, detta frequenza di Nyquist (o anche cadenza di Nyquist), necessaria per campionare un segnale
    #  analogico senza perdere informazioni
    nyq_rate = 1000 / 2

    # regione di transizione del filtro dove il filtro passa da attenuare a non attenuare
    width = 0.2 / nyq_rate

    # oscillazione del filtro
    ripple_db = 12

    # ordine del filtro, parametro del filtro ottenuto con pigreco * shape of the window
    num_of_taps, beta = kaiserord(ripple_db, width)
    if num_of_taps % 2 == 0:
        num_of_taps = num_of_taps + 1

    #  intervallo all'interno del quale attenuare
    cutoff_hz = np.array([59.5, 60.5])

    # filtro
    filter_bs = firwin(num_of_taps, cutoff_hz / nyq_rate, window=('kaiser', beta), pass_zero='bandstop')

    # valutare il filtro
    # w, h = freqz(filter_bs, worN=40000)
    #
    # plt.plot((w / np.pi) * nyq_rate, 20 * np.log10(np.abs(h)), linewidth=2)
    #
    # plt.axhline(-ripple_db, linestyle='--', linewidth=1, color='c')
    # delta = 10 ** (-ripple_db / 20)
    # plt.axhline(20 * np.log10(1 + delta), linestyle='--', linewidth=1, color='r')
    # plt.axhline(20 * np.log10(1 - delta), linestyle='--', linewidth=1, color='r')
    #
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Gain (dB)')
    # plt.title('Frequency Response')
    # plt.xlim(40, 80)
    # plt.grid(True)
    # plt.show()

    return filter_bs


def HighPassFilter():
    #  minima frequenza, detta frequenza di Nyquist (o anche cadenza di Nyquist), necessaria per campionare un segnale
    #  analogico senza perdere informazioni
    nyq_rate = 1000 / 2

    # regione di transizione del filtro dove il filtro passa da attenuare a non attenuare
    width = 0.2 / nyq_rate

    # oscillazione del filtro
    ripple_db = 12

    # ordine del filtro, parametro del filtro ottenuto con pigreco * shape of the window
    num_of_taps, beta = kaiserord(ripple_db, width)
    if num_of_taps % 2 == 0:
        num_of_taps = num_of_taps + 1

    #  frequenza al di sotto della quale attenuare
    cutoff_hz = 0.5

    # filtro
    filter_hf = firwin(num_of_taps, cutoff_hz / nyq_rate, window=('kaiser', beta), pass_zero='highpass')

    # valutare il filtro
    # w, h = freqz(filter_hf, worN=40000)
    #
    # plt.plot((w / np.pi) * nyq_rate, 20 * np.log10(np.abs(h)), linewidth=2)
    #
    # plt.axvline(cutoff_hz + width * nyq_rate, linestyle='--', linewidth=1, color='g')
    # plt.axvline(cutoff_hz - width * nyq_rate, linestyle='--', linewidth=1, color='g')
    # plt.axhline(-ripple_db, linestyle='--', linewidth=1, color='c')
    # delta = 10 ** (-ripple_db / 20)
    # plt.axhline(20 * np.log10(1 + delta), linestyle='--', linewidth=1, color='r')
    # plt.axhline(20 * np.log10(1 - delta), linestyle='--', linewidth=1, color='r')
    #
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Gain (dB)')
    # plt.title('Frequency Response')
    # plt.ylim(-40, 5)
    # plt.xlim(0, 5)
    # plt.grid(True)
    # plt.show()

    return filter_hf


def LowPassFilter():
    #  minima frequenza, detta frequenza di Nyquist (o anche cadenza di Nyquist), necessaria per campionare un segnale
    #  analogico senza perdere informazioni
    nyq_rate = 1000 / 2

    # regione di transizione del filtro dove il filtro passa da attenuare a non attenuare
    width = 1 / nyq_rate

    # oscillazione del filtro
    ripple_db = 12

    # ordine del filtro, parametro del filtro ottenuto con pigreco * shape of the window
    num_of_taps, beta = kaiserord(ripple_db, width)
    if num_of_taps % 2 == 0:
        num_of_taps = num_of_taps + 1

    #  frequenza al di sopra della quale attenuare
    cutoff_hz = 100

    # filtro
    filter_lp = firwin(num_of_taps, cutoff_hz / nyq_rate, window=('kaiser', beta), pass_zero='lowpass')

    # valutare il filtro
    # w, h = freqz(filter_lp, worN=40000)
    #
    # plt.plot((w / np.pi) * nyq_rate, 20 * np.log10(np.abs(h)), linewidth=2)
    #
    # plt.axvline(cutoff_hz + width * nyq_rate, linestyle='--', linewidth=1, color='g')
    # plt.axvline(cutoff_hz - width * nyq_rate, linestyle='--', linewidth=1, color='g')
    # plt.axhline(-ripple_db, linestyle='--', linewidth=1, color='c')
    # delta = 10 ** (-ripple_db / 20)
    # plt.axhline(20 * np.log10(1 + delta), linestyle='--', linewidth=1, color='r')
    # plt.axhline(20 * np.log10(1 - delta), linestyle='--', linewidth=1, color='r')
    #
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Gain (dB)')
    # plt.title('Frequency Response')
    # plt.ylim(-40, 5)
    # plt.xlim(0, 110)
    # plt.grid(True)
    # plt.show()

    return filter_lp


def SmoothSignal(signal):
    smoothed_signal = savgol_filter(signal, 29, 3)  # window size 51, polynomial order 3
    # plt.plot(signal[:int(len(signal) / 8)], color='k', label="raw signal")
    # plt.plot(smoothed_signal[:int(len(smoothed_signal) / 8)], label="smoothed signal")
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    return smoothed_signal
