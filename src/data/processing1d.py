import numpy as np
from scipy.signal import butter, sosfilt
from scipy.stats import zscore


def process1d(signal: np.ndarray) -> np.ndarray:
    """На одно отведение один столбец в массиве."""
    signal = butterworth_filter(signal)

    signal = znormolization(signal)

    return signal


band_pass_filter = butter(2, [1, 45], "bandpass", fs=100, output="sos")


def butterworth_filter(signal):
    """Band pass filter. Полосовой фильтр Баттерворта 2-ого порядка"""
    return sosfilt(band_pass_filter, signal, axis=0)


def znormolization(signal):
    """Z-score normalization"""
    return zscore(signal, axis=0)
