import os
import pickle
import numpy as np
import pandas as pd
import torch
import inspect
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

from .dataset import MyDataset
from .config_run import set_seed, set_device
from .metrics import compute_tptnfpfn, compute_all_tptnfpfn, compute_metrics, percentage
from .plotting import save_plot

project_path = os.getenv("project_root")


def start_training(
    model_class: type[nn.Module],
    dimension: int,
    model_name: str,
    dataset_name: str,
    parameters: dict,
    model_parameters: dict = {},
):
    """Основная функция запуска обучения и тестирования модели.
    Включает в себя:

    Args:
        model_class (type[nn.Module]): класс модели
        dimension (int): размерность входных данных
        model_name (str): имя, под которым будет сохранена обученная модель
        dataset_name (str): имя датасета, из которого будут загружены данные
        parameters (dict): гиперпараметры обучения в виде словаря (epochs, batch_size, learning_rate, l2_decay, early_stop, optimizer: str = ["adam", "sgd"], device: str = ["cuda", "cpu", "mps"])
        model_parameters (dict): Параметры класса модели
    """

    # meta.keys() = (n_sig, fs, n_classes, labels, n_MI, n_STTC, n_CD, n_HYP, n_train_dev_test, n_train, n_dev, n_test)
    device = parameters["device"]
    meta = Path(
        project_path, "data", "processed", f"{dimension}D", dataset_name, "meta.pkl"
    )
    with open(meta, "rb") as f:
        meta = pickle.load(f)

    # установка устройства, на котором запускается обучение
    set_device(device)
    # установка seed на все модули рандома
    set_seed(2024)

    model = model_class(**model_parameters).to(device)
    dataset_train = MyDataset(dataset_name, dimension, "train", meta["n_train"])
    dataset_dev = MyDataset(dataset_name, dimension, "dev", meta["n_dev"])
    dataset_test = MyDataset(dataset_name, dimension, "test", meta["n_test"])

    dataloader_train = DataLoader(
        dataset_train, batch_size=parameters["batch_size"], shuffle=True
    )
    dataloader_dev = DataLoader(dataset_dev, batch_size=1, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

    optimizer = (
        torch.optim.Adam if parameters["optimizer"] == "adam" else torch.optim.SGD
    )
    optimizer = optimizer(
        model.parameters(),
        lr=parameters["learning_rate"],
        weight_decay=parameters["l2_decay"],
    )

    # https://discuss.pytorch.org/t/weighted-binary-cross-entropy/51156/6
    # https://discuss.pytorch.org/t/multi-label-multi-class-class-imbalance/37573/2
    labels_weights = torch.tensor(
        [
            meta["n_train_dev_test"] / meta["n_MI"],
            meta["n_train_dev_test"] / meta["n_STTC"],
            meta["n_train_dev_test"] / meta["n_CD"],
            meta["n_train_dev_test"] / meta["n_HYP"],
        ],
        dtype=torch.float,
    )
    loss_function = nn.BCEWithLogitsLoss(pos_weight=labels_weights).to(device)
    # https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
    # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html

    # эпохи в виде массива, для дальнейшего plot
    epochs = np.arange(1, parameters["epochs"] + 1)
    statistics = pd.DataFrame(
        data=np.zeros((parameters["epochs"], 4)),
        index=epochs,
        columns=("Training-loss", "Validation-loss", "Sensitivity", "Specificity"),
    )
    # потери за каждый батч
    train_losses = np.zeros(int(np.ceil(meta["n_train"] / dataloader_train.batch_size)))

    best_model_sensitivity = (model, 0)
    best_model_dev_loss = (model, float("inf"))
    patience_i = 0
    patience_limit = parameters["patience_limit"]
    for epoch in epochs:
        print(f"Epoch {epoch}")

        train_losses[:] = 0
        for i, (X, label) in enumerate(dataloader_train):
            loss = train_batch(X, label, model, optimizer, loss_function, device)

            del X, label
            torch.cuda.empty_cache()

            train_losses[i] = loss

        train_mean_loss = train_losses.mean().item()
        dev_mean_loss = compute_dev_loss(
            model, dataloader_dev, loss_function, meta["n_dev"], device
        )
        statistics.at[epoch, "Training-loss"] = train_mean_loss
        statistics.at[epoch, "Validation-loss"] = dev_mean_loss
        quality_metrics = evaluate(
            model, dataloader_dev, meta["labels"].values(), device
        )
        statistics.at[epoch, "Sensitivity"] = quality_metrics.at["all", "Sensitivity"]
        statistics.at[epoch, "Specificity"] = quality_metrics.at["all", "Specificity"]

        print(f"Training-Loss: {train_mean_loss}")
        print(f"Validation-Loss: {dev_mean_loss}")
        print(f"Sensitivity: {quality_metrics.at["all","Sensitivity"]:.4f}")
        print(f"Specificity: {quality_metrics.at["all", "Specificity"]:.4f}")

        if quality_metrics.at["all", "Specificity"] > best_model_sensitivity[1]:
            best_model_sensitivity = (model, quality_metrics.at["all", "Specificity"])
        if dev_mean_loss < best_model_dev_loss[1]:
            best_model_dev_loss = (model, dev_mean_loss)
            patience_i = 0
        else:
            patience_i += 1

        if patience_i == patience_limit:
            print("Early stopping")
            break

    # Тестирование на тестовой выборке
    test_quality_metrics = evaluate(
        best_model_dev_loss[0],
        dataloader_test,
        meta["labels"].values(),
        device,
    )

    # Сохраняем модель
    path_save_model = Path(project_path, "models", model_name)
    path_save_model.mkdir(parents=True, exist_ok=True)
    torch.save(
        best_model_sensitivity[0].state_dict(),
        path_save_model / f"{model_name}_max_sens.pth",
    )
    torch.save(
        best_model_dev_loss[0].state_dict(),
        path_save_model / f"{model_name}_min_loss.pth",
    )

    # Сохраняем результаты
    path_reports = Path(project_path, "reports", model_name)
    path_reports.mkdir(parents=True, exist_ok=True)

    statistics.to_html(
        path_reports / "training_report.html",
        index=True,
        col_space=100,
        float_format=lambda x: f"{x:.4f}",
        justify="left",
    )

    test_quality_metrics = percentage(test_quality_metrics)
    test_quality_metrics.to_html(
        path_reports / "test_report.html",
        index=True,
        col_space=100,
        float_format=lambda x: f"{x}%",
        justify="left",
    )
    save_plot(
        statistics["Validation-loss"][statistics["Validation-loss"] != 0],
        "Validation-Loss",
        path_reports,
        "validation_loss",
    )
    save_plot(
        statistics["Training-loss"][statistics["Training-loss"] != 0],
        "Training-Loss",
        path_reports,
        "training_loss",
    )
    save_plot(
        statistics["Sensitivity"][statistics["Sensitivity"] != 0],
        "Sensitivity",
        path_reports,
        "sensitivity",
    )
    save_plot(
        statistics["Specificity"][statistics["Specificity"] != 0],
        "Specificity",
        path_reports,
        "specificity",
    )

    # Сохраняем параметры запуска модели
    with open(path_reports / "parameters.txt", "w") as f:
        if not model_parameters:
            model_parameters = {
                k: value.default
                for k, value in inspect.signature(model_class).parameters.items()
            }
        print(
            f"parameters\n{parameters}",
            f"model_parameters\n{model_parameters}",
            model_parameters,
            sep="\n\n",
            file=f,
        )


def train_batch(
    X: torch.Tensor,
    label: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.BCEWithLogitsLoss,
    device: str,
) -> float:
    X, label = X.to(device), label.to(device)
    # Обнуляем градиенты всех параметров
    optimizer.zero_grad()
    # Вызов метода forward(). Прямой проход по сети
    out = model(X)
    # Вычисление функции потерь. criterion - функция потерь
    loss = loss_function(out, label)
    # Обратное распространение функции потерь. Вычисление градиентов функции потерь
    loss.backward()
    # Обновление параметров оптимизатором на основе вычисленных ранее градиентов
    optimizer.step()
    # Возвращаем значение функции потерь
    return loss.item()


def compute_dev_loss(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.BCEWithLogitsLoss,
    n_dev: int,
    device: str,
) -> float:
    """Вычисление потерь на валидационном датасете

    Args:
        model (nn.Module): модель
        dataloader (DataLoader): датасет
        loss_function (nn.BCEWithLogitsLoss): функция потерь
        n_int (int): количество образцов в валидационном датасете
        device (str): устройство "cpu" или "cuda" или "mps"
    Returns:
        float: потери
    """
    model.eval()
    with torch.no_grad():
        dev_losses = np.zeros(int(np.ceil(n_dev / dataloader.batch_size)))
        for i, (X, label) in enumerate(dataloader):
            X, label = X.to(device), label.to(device)
            y_pred = model(X)
            loss = loss_function(y_pred, label)
            dev_losses[i] = loss.item()

            del X, label
            torch.cuda.empty_cache()

    model.train()

    return np.mean(dev_losses)


def predict(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    """Предсказание модели

    Args:
        model (nn.Module): модель
        X (torch.Tensor): (batch_size, ...)

    Returns:
        np.ndarray: (batch_size, n_classes)
    """
    # logits_ - логиты (ненормализованные вероятности) для каждого класса для каждого примера в пакете данных.
    logits = model(X)  # (batch_size, n_classes)
    # sigmoid - преобразование логитов в вероятности, т.е. в числа в диапазоне [0,1]
    probabilities = torch.sigmoid(logits).cpu()

    return (probabilities > 0.5).numpy()


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    diagnostic_classes: tuple[str],
    device: str,
) -> pd.DataFrame:
    """Оценка качества модели

    Args:
        model (nn.Module): модель
        dataloader (DataLoader): dev_dataloader or test_dataloader
        diagnostic_classes (tuple[str]): классы диагнозов meta["labels"] = {0: "MI", 1: "STTC", 2: "CD", 3: "HYP"}
        device (str): "cpu" or "cuda" or "mps"

    Returns:
        pd.DataFrame:
        |      | TP | TN | FP | FN | Sensitivity | Specificity | G-mean |
        |------|----|----|----|----|-------------|-------------|--------|
        | MI   |    |    |    |    |             |             |        |
        | STTC |    |    |    |    |             |             |        |
        | CD   |    |    |    |    |             |             |        |
        | HYP  |    |    |    |    |             |             |        |
        | all  |    |    |    |    |             |             |        |
    """
    # Перевод модели в режим оценки
    model.eval()
    # Отключение вычисления градиентов
    with torch.no_grad():
        diagnostic_classes = tuple(diagnostic_classes) + ("all",)
        quality_metrics = pd.DataFrame(
            data=np.zeros((len(diagnostic_classes), 7)),
            index=diagnostic_classes,
            columns=("TP", "TN", "FP", "FN", "Sensitivity", "Specificity", "G-mean"),
        )
        for i, (X, label) in enumerate(dataloader):
            X, label = X.to(device), label.to(device)
            # прогноз модели
            y_predict = predict(model, X)
            # метки
            y_true = label.cpu().numpy()

            quality_metrics = compute_tptnfpfn(y_predict, y_true, quality_metrics)

            # Очистка памяти
            del X, label
            torch.cuda.empty_cache()
    quality_metrics = compute_all_tptnfpfn(quality_metrics)
    quality_metrics = compute_metrics(quality_metrics)
    # Возврат модели в режим обучения
    model.train()

    return quality_metrics
