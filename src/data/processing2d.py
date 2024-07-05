import numpy as np
import torch
from scipy.signal import butter, sosfilt
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
from torchvision.models import AlexNet_Weights


def process2d(signal: np.ndarray) -> np.ndarray:
    signal = butterworth_filter(signal)

    signal = minmax_normalization(signal)
    imgs = transform_to_image(signal)
    imgs = resize(imgs)

    return imgs


band_pass_filter = butter(2, [1, 45], "bandpass", fs=100, output="sos")


def butterworth_filter(signal):
    """Band pass filter. Полосовой фильтр Баттерворта 2-ого порядка"""
    return sosfilt(band_pass_filter, signal, axis=0)


def minmax_normalization(signal: np.ndarray) -> np.ndarray:
    return (signal - np.min(signal, axis=0)) / (
        np.max(signal, axis=0) - np.min(signal, axis=0)
    )


GAF = GramianAngularField(image_size=1000, method="summation")
RP = RecurrencePlot(threshold="distance", percentage=20)
MTF = MarkovTransitionField(image_size=1000, n_bins=20)


def transform_to_image(signal: np.ndarray) -> np.ndarray:
    # signal.shape() = (длина отведений = 1000 или 5000, кол-во отведений)
    # 3 изображения GAF, RP, MTF
    imgs = np.zeros((signal.shape[1] * 3, signal.shape[0], signal.shape[0]))

    i = 0
    for sig in signal.T:
        sig = sig.reshape(1, -1)
        imgs[i] = GAF.transform(sig)
        imgs[i + 1] = RP.transform(sig)
        imgs[i + 2] = MTF.transform(sig)
        i += 3

    return imgs


transform_resize = AlexNet_Weights.IMAGENET1K_V1.transforms()


def resize(imgs: np.ndarray) -> np.ndarray:
    return (
        transform_resize(torch.from_numpy(imgs.reshape(3, 3, 1000, 1000)))
        .numpy()
        .reshape(9, 224, 224)
    )
