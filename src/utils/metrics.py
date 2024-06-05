import numpy as np
import pandas as pd


def compute_tptnfpfn(
    y_predict: np.ndarray, y_true: np.ndarray, quality_metrics: pd.DataFrame
):
    """Вычисление метрик TP | TN | FP | FN для каждого класса в метке.

    Args:
        y_predict (np.ndarray): прогноз модели (batch_size, num_classes)
        y_true (np.ndarray): метка (batch_size, num_classes)
        quality_metrics (pd.DataFrame): Таблица вида

        |      | TP | TN | FP | FN | Sensitivity | Specificity | G-mean |
        |------|----|----|----|----|-------------|-------------|--------|
        | MI   |    |    |    |    |             |             |        |
        | STTC |    |    |    |    |             |             |        |
        | CD   |    |    |    |    |             |             |        |
        | HYP  |    |    |    |    |             |             |        |
        | all  |    |    |    |    |             |             |        |
    Returns:
        pd.DataFrame: обновленный quality_metrics
    """
    for prediction, true in zip(y_predict, y_true):
        for i in range(len(prediction)):
            if prediction[i] == 1 and true[i] == 1:
                quality_metrics.iat[i, 0] += 1
            elif prediction[i] == 0 and true[i] == 0:
                quality_metrics.iat[i, 1] += 1
            elif prediction[i] == 1 and true[i] == 0:
                quality_metrics.iat[i, 2] += 1
            elif prediction[i] == 0 and true[i] == 1:
                quality_metrics.iat[i, 3] += 1

    return quality_metrics


def compute_all_tptnfpfn(quality_metrics: pd.DataFrame):
    """Вычисление метрик общего кол-ва TP | TN | FP | FN (строка 'all').

    Args:
        quality_metrics (pd.DataFrame): Таблица вида
        |      | TP | TN | FP | FN | Sensitivity | Specificity | G-mean |
        |------|----|----|----|----|-------------|-------------|--------|
        | MI   |    |    |    |    |             |             |        |
        | STTC |    |    |    |    |             |             |        |
        | CD   |    |    |    |    |             |             |        |
        | HYP  |    |    |    |    |             |             |        |
        | all  |    |    |    |    |             |             |        |
    Returns:
        pd.DataFrame: обновленный quality_metrics
    """
    for column in quality_metrics.columns:
        quality_metrics.at["all", column] = quality_metrics[column].sum()

    return quality_metrics


def compute_metrics(quality_metrics: pd.DataFrame):
    """Вычисление метрик Sensitivity | Specificity | G-mean для каждого класса в метке.

    Args:
        quality_metrics (pd.DataFrame): Таблица вида

        |      | TP | TN | FP | FN | Sensitivity | Specificity | G-mean |
        |------|----|----|----|----|-------------|-------------|--------|
        | MI   |    |    |    |    |             |             |        |
        | STTC |    |    |    |    |             |             |        |
        | CD   |    |    |    |    |             |             |        |
        | HYP  |    |    |    |    |             |             |        |
        | all  |    |    |    |    |             |             |        |
    Returns:
        pd.DataFrame: обновленный quality_metrics
    """
    quality_metrics["Sensitivity"] = quality_metrics["TP"] / (
        quality_metrics["TP"] + quality_metrics["FN"]
    )
    quality_metrics["Specificity"] = quality_metrics["TN"] / (
        quality_metrics["TN"] + quality_metrics["FP"]
    )
    quality_metrics["G-mean"] = np.sqrt(
        (quality_metrics["Sensitivity"] + quality_metrics["Specificity"])
    )

    return quality_metrics
