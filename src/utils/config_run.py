import random
import numpy as np
import torch


class DeviceError(Exception):
    pass


def set_seed(seed: int):
    """
    Конфигурация случайного генератора для воспроизводимости результатов обучения
    Устанавливает seed для всех используемых модулей:
    - random
    - numpy
    - torch
    - torch.cuda

    Также устанавливает флаг torch.backends.cudnn.deterministic = True также для воспроизводимости результатов обучения. В обычном режиме (при False) работы cuDNN использует оптимизированные алгоритмы, которые могут включать в себя некоторую степень неопределенности в вычислениях.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


def set_device(device: str):
    if device == "cpu":
        torch.device("cpu")
        return

    if device == "cuda" and torch.cuda.is_available():
        torch.device("cuda")
        return

    if device == "mps" and torch.backends.mps.is_available():
        torch.device("mps")
        return

    raise DeviceError(
        "The specified device is not available. Available devices: 'cpu', 'cuda', 'mps'"
    )
