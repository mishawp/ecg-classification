import wfdb
import numpy as np
import pandas as pd
import ast
import os
import pickle
from typing import Any
from pathlib import Path
from scipy.signal import butter, sosfilt
from scipy.stats import zscore
from .processing1d import process1d
from .processing2d import process2d


def make_dataset(
    raw_dataset_dir_name: str,
    processed_dataset_dir_name: str,
    dimension: int,
    sampling_rate: int,
    leads: list[str],
    bar: Any = None,
):
    PROJECT_ROOT = Path(os.getenv("project_root"))
    PATH_RAW_DATA = Path(
        PROJECT_ROOT, "data", "raw", raw_dataset_dir_name
    )  # путь к каталогу с датасетом
    PATH_PROCESSED_DATA = Path(
        PROJECT_ROOT, "data", "processed", f"{dimension}D", processed_dataset_dir_name
    )  # путь к каталогу с обработанным датасетом
    for part in ("train", "dev", "test"):
        Path(PATH_PROCESSED_DATA, f"X_{part}").mkdir(parents=True, exist_ok=True)
        Path(PATH_PROCESSED_DATA, f"labels_{part}").mkdir(parents=True, exist_ok=True)

    DEV_FOLD = 9
    TEST_FOLD = 10

    # load and convert annotation data (см. project/notebooks/1.0-vmr-ptbxl-data-review.ipynb)
    Y = pd.read_csv(Path(PATH_RAW_DATA, "ptbxl_database.csv"), index_col="ecg_id")
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load scp_statements.csv for diagnostic aggregation
    scp_statements = pd.read_csv(Path(PATH_RAW_DATA, "scp_statements.csv"), index_col=0)
    scp_statements = scp_statements[scp_statements.diagnostic == 1]

    # Добавляем поле diagnostic_class к каждой записи из БД. diagnostic_class - это класс, обозначающий группу болезней.
    Y["diagnostic_class"] = Y.scp_codes.apply(
        lambda scp_codes: list(
            {
                scp_statements.loc[diagnostic].diagnostic_class
                for diagnostic in scp_codes
                if diagnostic in scp_statements.index
            }
        )
    )
    DIAGNOSTIC_CLASSES = list(
        set(scp_statements.diagnostic_class.values)
    )  # метки (классы болезней)
    LABELS_COUNT = (
        len(DIAGNOSTIC_CLASSES) - 1
    )  # количество меток (классов болезней) . Norm - все 0, поэтому -1

    i_train, i_dev, i_test = 0, 0, 0
    n_MI, n_STTC, n_CD, n_HYP = (
        0,
        0,
        0,
        0,
    )
    total = len(Y) + 20
    current = 0
    filename = "filename_hr" if sampling_rate == 500 else "filename_lr"
    for y in Y.itertuples():
        signal, meta = wfdb.rdsamp(Path(PATH_RAW_DATA, getattr(y, filename)))
        signal = signal[:, [meta["sig_name"].index(lead) for lead in leads]]

        if dimension == 1:
            signal = process1d(signal)
        elif dimension == 2:
            signal = process2d(signal)

        # Представим метки в виде вектора 0 и 1
        labels = np.zeros(LABELS_COUNT)
        for label in y.diagnostic_class:
            # labels[diagnostic_classes.index(label)] = 1  # one-hot encoding
            if "MI" in label:
                labels[0] = 1
                n_MI += 1
            if "STTC" in label:
                labels[1] = 1
                n_STTC += 1
            if "CD" in label:
                labels[2] = 1
                n_CD += 1
            if "HYP" in label:
                labels[3] = 1
                n_HYP += 1

        # Разбиение на тренировочную, валидационную и тестовую выборки
        if y.strat_fold == DEV_FOLD:
            np.save(Path(PATH_PROCESSED_DATA, "X_dev", f"{i_dev}.npy"), signal)
            np.save(Path(PATH_PROCESSED_DATA, "labels_dev", f"{i_dev}.npy"), labels)
            i_dev += 1
        elif y.strat_fold == TEST_FOLD:
            np.save(Path(PATH_PROCESSED_DATA, "X_test", f"{i_test}.npy"), signal)
            np.save(Path(PATH_PROCESSED_DATA, "labels_test", f"{i_test}.npy"), labels)
            i_test += 1
        else:
            np.save(Path(PATH_PROCESSED_DATA, "X_train", f"{i_train}.npy"), signal)
            np.save(Path(PATH_PROCESSED_DATA, "labels_train", f"{i_train}.npy"), labels)
            i_train += 1

        current += 1
        if bar is not None:
            bar(current, total)

    # некоторые необходимые дополнительные данные
    save_meta = {
        "n_sig": 3,
        "fs": sampling_rate,
        "n_classes": LABELS_COUNT,
        "labels": {0: "MI", 1: "STTC", 2: "CD", 3: "HYP"},
        "n_MI": n_MI,
        "n_STTC": n_STTC,
        "n_CD": n_CD,
        "n_HYP": n_HYP,
        "n_train_dev_test": i_train + i_dev + i_test,
        "n_train": i_train,
        "n_dev": i_dev,
        "n_test": i_test,
    }

    with open(Path(PATH_PROCESSED_DATA, "meta.pkl"), "wb") as outfile:
        pickle.dump(save_meta, outfile)

    if bar is not None:
        bar(current + 20, total)
