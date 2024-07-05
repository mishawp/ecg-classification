ecg-classification
==============================

Automation of disease diagnosis based on electrocardiogram data based on machine learning algorithms (Neural Networks)

Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── .env               <- переменные окружения. Должен содержать project_root='path', где path - путь к корню проекта.
    │
    ├── src                <- Source code for use in this project.
    │   │
    │   ├── data           <- Scripts to download and generate data
    │   │
    │   ├── GUI            <- Графический интерфейс, который объединяет весь остальной код
    │   │
    │   ├── nn1d           <- Одномерные нейронные сети
    │   │
    │   ├── nn2d           <- Двумерные нейронные сети
    │   │
    │   ├── utils          <- Функции и классы: загрузчики данных в NN, рисование графиков, таблиц, архитектур NN
    │   │
    │   ├── main.py        <- Запускает графический интерфейс

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
