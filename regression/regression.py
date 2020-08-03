"""
This script perfoms the basic process for applying a machine learning
algorithm to a dataset using Python libraries.

The four steps are:
   1. Download a dataset (using pandas)
   2. Process the numeric data (using numpy)
   3. Train and evaluate learners (using scikit-learn)
   4. Plot and compare results (using matplotlib)


The data is downloaded from URL, which is defined below.

============
Example Data
============
The example is from https://web.archive.org/web/20180322001455/http://mldata.org/repository/data/viewslug/stockvalues/
It contains stock prices and the values of three indices for each day
over a five year period. See the linked page for more details about
this data set.

This script uses regression learners to predict the stock price for
the second half of this period based on the values of the indices. This
is a naive approach, and a more robust method would use each prediction
as an input for the next, and would predict relative rather than
absolute values.
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_table
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split


def download_data(path, separator, index_column, header_row, target_columns):
    """
    Завантажує дані для скрипту у формат pandas DataFrame.
    """

    frame = read_table(
        path,

        # Визначити кодування файлу
        encoding='utf-8',

        # Визначити роздільник даних
        sep=separator,

        # Ігнорує пробіли після оздільника
        skipinitialspace=True,

        # Обрати індекси для рядків
        index_col=index_column,

        # Обрати заголовки стовпців
        header=header_row

    )

    return frame[target_columns]


def get_features_and_labels(frame, target_value_column, missed_value_policy, data_percent=0.5):
    """
    Перетворює та масштабує вхідні дані та повертає numpy масиви для
     навчання та тестування вхідних даних та цільових значень (міток).
    """

    # Пропущені значення можуть бути заміені нулями
    if missed_value_policy == 0:
        frame[frame.isnull()] = 0.0

    # Конвертування даних у масив типу float
    arr = np.array(frame, dtype=np.float)

    # Обрати стовпець з міками даних
    if target_value_column == 1:
        X, y = arr[:, 1:], arr[:, 0]
    else:
        X = arr[:, :-1]
        y = arr[:, -1]

    # Визначення даних для тренування
    X_train, _, y_train, _ = train_test_split(X, y, test_size=data_percent)
    X_test, y_test = X, y

    # Пропущені занчення можуть бути замінені даними з набору
    if missed_value_policy == 1:
        imputer = SimpleImputer(strategy='mean')
        imputer.fit(X_train)
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)

    # Нормалізування значення атрибутів для значення = 0 і дисперсії = 1
    scaler = StandardScaler()

    # Налаштування масштабування з тренувальних даних
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def evaluate_learner(X_train, X_test, y_train, y_test):
    """
    Запускається кілька разів з різними алгоритмами, щоб отримати уявлення про
     відносна продуктивність кожної конфігурації.

     Повертає послідовність кортежів, що містять:
         (назва, очікувані значення, фактичні значення)
     для кожного учня.
    """

    # Тренування методом линійних опорних векторів T
    svr = SVR(kernel='linear')
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    r_2 = svr.score(X_test, y_test)
    yield 'Лінійна модель ($Якість={:.3f}$)'.format(r_2), y_test, y_pred

    # Тренування поліноміальної моделі
    svr = SVR(kernel='poly')
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    r_2 = svr.score(X_test, y_test)
    yield 'Поліноміальна модель ($Якість={:.3f}$)'.format(r_2), y_test, y_pred

    # Тренування з використанням ядерної моделі
    svr = SVR(kernel='rbf')
    svr.fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    r_2 = svr.score(X_test, y_test)
    yield 'RBF модель ($Якість={:.3f}$)'.format(r_2), y_test, y_pred


def plot(results, path, data_percent):
    """
    Створює графіки для порівняння кількох алгоритмів.

    `результат` цес список is a список кортежів, що містять:
        (заголовок, очікувані значення, фактичні значення)

    """

    # Використовується subplots для відображення результатів в одній осі Х
    fig, plts = plt.subplots(nrows=len(results), figsize=(8, 8))
    fig.canvas.set_window_title('Передбачення даних з ' + path)

    # Відображає кожний графік, отриманий з plt.subplots()
    for subplot, (title, y, y_pred) in zip(plts, results):
        # Назва вертикальної осі
        subplot.set_ylabel('Ціни акцій')

        # Виставити назву графіку
        subplot.set_title(title)

        # Друк отриманих даних та передбачень
        subplot.plot(y, 'b', label='Отримані дані')
        subplot.plot(y_pred, 'r', label='Передбачені дані')

        # Відмітити навчальні дані
        subplot.axvline(len(y) * data_percent, linestyle='--', color='0', alpha=0.2)

        # Добавити легенду на граіки
        subplot.legend()

    fig.tight_layout()

    # Друг графіків
    plt.show()

    # Звільнити пам'ять після закриття графіків
    plt.close()


def analyze(app, path, settings):
    # Вилучення налаштувань з кортежу
    separator = settings[0]
    index_column = settings[1]
    header_row = settings[2]
    target_value_column = settings[3]
    missed_value_policy = settings[4]
    data_percent = settings[5]
    target_columns = settings[6]

    # Завантаження набору даних за шляхом
    app.print_logs("Завантаження даних з {}...".format(path))
    frame = download_data(path, separator, index_column, header_row, target_columns)

    # Обробка даних у масиви властивостей та міток
    app.print_logs("Обробка {} зразків з {} атрибутами...".format(len(frame.index), len(frame.columns)))
    X_train, X_test, y_train, y_test = get_features_and_labels(frame, target_value_column, missed_value_policy, data_percent)

    # Оцінювання декількох алгоритмів навчання регресії за даними
    app.print_logs("Оцінювання навчання регресії...")
    results = list(evaluate_learner(X_train, X_test, y_train, y_test))

    app.print_logs("Друк результатів...")
    plot(results, path, data_percent)
