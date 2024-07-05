import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def save_plot(x_y, ylabel, path, name):
    """сохранение графика

    Args:
        x_y: двумерный итерабельный объект
        ylabel (str): имя оси ордина
        path (str | Path): путь к папке для сохранения
        name (str): имя файла сохранения
    """
    plt.clf()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.plot(x_y)
    plt.savefig(Path(path, f"{name}.png"), bbox_inches="tight")


def visualize(model, path, name):
    pass
