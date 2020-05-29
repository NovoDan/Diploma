"""

The four steps are:
   1. Download a dataset (using pandas)
   2. Process the numeric data (using numpy)
   3. Train and evaluate learners (using scikit-learn)
   4. Plot and compare results (using matplotlib)

============
Example Data
============
The example is from http://mlr.cs.umass.edu/ml/datasets/Spambase
It contains pre-processed metrics, such as the frequency of certain
words and letters, from a collection of emails. A classification for
each one indicating 'spam' or 'not spam' is in the final column.
See the linked page for full details of the data set.

This script uses three classifiers to predict the class of an email
"""

import numpy as np
import matplotlib.pyplot as plt
from pandas import read_table
from sklearn.svm import LinearSVC, NuSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# =====================================================================


def download_data(path, separator, index_column, header_row):
    """
    Завантажує дані для скрипту у формат pandas DataFrame.
    """

    frame = read_table(
        path,

        # Визначає кодування файлу
        encoding='utf-8',

        # Визначає роздільник даних
        sep=separator,

        # Ігнорує пробіли після оздільника
        skipinitialspace=True,

        # Обрати індекси для рядків
        index_col=index_column,

        # Обрати заголовки стовпців
        header=header_row

    )

    # Return the entire frame
    return frame


# =====================================================================


def get_features_and_labels(frame, target_value_column, missed_value_policy, data_percent):
    """
    Перетворює та масштабує вхідні дані та повертає numpy масиви для
     навчання та тестування вхідних даних та цільових значень (міток).
    """

    arr = np.array(frame)

    # Обрати стовпець з міками даних
    if target_value_column == 1:
        X, y = arr[:, 1:], arr[:, 0]
    else:
        X = arr[:, :-1]
        y = arr[:, -1]

    # Пропущені значення можуть бути заміені нулями
    if missed_value_policy == 0:
        frame[frame.isnull()] = 0.0

    # Розділяє дані на тренувальні та тестові
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data_percent)

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


# =====================================================================


def evaluate_classifier(X_train, X_test, y_train, y_test):
    """
    Запускається декілька разів з різними клисифікаторами, щоб отримати уявлення про
     відносна продуктивність кожної конфігурації.

     Повертає послідовність кортежів, що містять:
         (назва, точність, повнота, колір графіку)
     для кожного з алгоритмів що навчаються.
    """

    # Створюється крласифікатор з параметрами за змовчуванням.
    # Вони повинні бути відрегульовані для отримання оптимальної продуктивності набору даних.

    # Test the Nu support vector classifier
    classifier = NuSVC(kernel='rbf', nu=0.5, gamma=1e-3)
    # Навчання класифікатора
    classifier.fit(X_train, y_train)
    # Розрахування середньозваженого 'точність-повнота'
    score = f1_score(y_test, classifier.predict(X_test))
    # Генерування кривої повнота-точність
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    yield 'Метод опорних векторів (SVC) (якість={:.3f})'.format(score), precision, recall, 'yellow'

    # Тестування метода опроних векторів
    classifier = LinearSVC(C=1)
    # Навчання класифікатора
    classifier.fit(X_train, y_train)
    # Розрахування середньозваженого 'точність-повнота'
    score = f1_score(y_test, classifier.predict(X_test))
    # Генерування кривої повнота-точність
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    yield 'Лінійний метод опорних векторів (LinearSVC) (якість={:.3f})'.format(score), precision, recall, 'red'

    # Test the Ada boost classifier
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')
    # Навчання класифікатора
    classifier.fit(X_train, y_train)
    # Розрахування середньозваженого 'точність-повнота'
    score = f1_score(y_test, classifier.predict(X_test))
    # Генерування кривої повнота-точність
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    yield 'Ada Boost (якість={:.3f})'.format(score), precision, recall, 'black'


# =====================================================================


def plot(results, URL):
    """
    Створює графіки для порівняння кількох алгоритмів.

     `Результати` - це список кортежів, що містять:
         (назва, точність, повнота)

    """

    # Друк кривих 'точність-повнота'

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('Класифікація даних з ' + URL)

    for label, precision, recall, color in results:
        plt.plot(recall, precision, label=label, color=color)

    # Встановити назви графіку, осей та розташування легенди
    plt.title('Криві Точність-Повнота')
    plt.xlabel('Точність')
    plt.ylabel('Повнота')
    plt.legend(loc='lower left')

    # Дати matplotlib покращити макет
    plt.tight_layout()

    # ==================================
    # Відобразити графіки
    plt.show()

    # Звільнити пам'ять після закриття графіків
    plt.close()


# =====================================================================

def analyze(app, path, settings):
    #Вилучення налаштувань з кортежу
    separator = settings[0]
    index_column = settings[1]
    header_row = settings[2]
    target_value_column = settings[3]
    missed_value_policy = settings[4]
    data_percent = settings[5]

    app.print_logs("Завантаження даних з {}...".format(path))
    try:
        frame = download_data(path, separator, index_column, header_row)
    except:
        app.print_logs('Помилка! Некорректні віхідні дані або налаштування!')
        return

    # Обробка даних у масиви властивостей да міток
    app.print_logs("Обробка {} зразків з {} атрибутами...".format(len(frame.index), len(frame.columns)))
    X_train, X_test, y_train, y_test = get_features_and_labels(frame, target_value_column, missed_value_policy, data_percent)

    # Оцінка класифікаторів на тренувальних та тестових даних
    app.print_logs("Оцінювання класифікаторів...")
    results = list(evaluate_classifier(X_train, X_test, y_train, y_test))

    # Відобразити результати
    app.print_logs("Друк результатів...")
    plot(results, path)
