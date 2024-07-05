"""Скачивание датасета из интернета и распаковка его в каталог ./data/raw"""

import os
import zipfile
import urllib
import wget
from pathlib import Path
from typing import Any
from functools import wraps


# def adaptive_bar(func) -> Any:
#     @wraps(func)
#     def wrapper()


def fetch_ecg_data(
    url: Path | str, path_save: Path | str, bar: Any = wget.bar_adaptive
):
    """Скачивает датасет в формате zip и распаковывает его в указанный каталог. Создает zip каталог (data.zip), и автоматически его удаляет после распаковки. Ничего не делает, если указанный в параметрах каталог уже существует."""
    zip_file = Path(path_save, "data.zip")
    # Создаем каталог для хранения датасета
    os.makedirs(path_save, exist_ok=True)
    # Скачивает архив с датасет в указанный файл
    wget.download(str(url), str(zip_file), bar)
    # Открытие архива и распаковка архива
    with zipfile.ZipFile(zip_file, "r") as z:
        z.extractall(path_save)
    os.remove(zip_file)
    return True
    # print("Done")


if __name__ == "__main__":
    download_root = r"https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"

    project_dir = Path(__file__).resolve().parents[2]
    local_path = os.path.join(project_dir, "data", "raw")
    fetch_ecg_data(url=download_root, path_save=local_path)
