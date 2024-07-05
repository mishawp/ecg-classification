import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

PROJECT_ROOT = os.getenv("project_root")


class MyDataset(Dataset):
    def __init__(self, dataset_name: str, dimension: int, part: str, n_samples: int):
        """Интерфейс датасета

        Args:
            dataset_name (str): название датасета
            dimension (int): размерность входных данных
            part (str): часть датасета (train, dev, test)
            n_samples (int): количество образцов
        """
        self.dataset_name = dataset_name
        self.dimension = dimension
        self.part = part
        self.n_samples = n_samples

        self.path_data_X_part = Path(
            PROJECT_ROOT,
            "data",
            "processed",
            f"{dimension}D",
            dataset_name,
            f"X_{part}",
        )
        self.path_data_labels_part = Path(
            PROJECT_ROOT,
            "data",
            "processed",
            f"{dimension}D",
            dataset_name,
            f"labels_{part}",
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        X = np.load(Path(self.path_data_X_part, f"{idx}.npy"))
        label = np.load(Path(self.path_data_labels_part, f"{idx}.npy"))

        return torch.tensor(X).float(), torch.tensor(label).float()
