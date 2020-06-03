import sys
from PyQt5 import QtWidgets
import ui.pydesign as design
from classifier import classifier
from regression import regression
from clustering import clustering


# python -m PyQt5.uic.pyuic -x pydesign.ui -o pydesign.py


class MachineLearningApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    # Конструктор
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # Ініціалізація дизайну
        # Пошук файлів
        self.btn_browse_classification.clicked.connect(
            self.browse_file_classifier)  # Прив'язати функцію browse_file до кнопки
        self.btn_browse_regression.clicked.connect(self.browse_file_regression)
        # Завантаження даних
        self.btn_download_classifier_data.clicked.connect(self.process_classifier_data)
        self.btn_download_regression_data.clicked.connect(self.process_regression_data)
        self.btn_start_clustering.clicked.connect(self.start_clustering)
        self.checkBox_kmeans.clicked.connect(self.enable_kmeans_settings)
        self.checkBox_dbscan.clicked.connect(self.enable_dbscan_settings)

    # Класифікація ======================================================

    # Отримання шляху до файлу
    def browse_file_classifier(self):
        self.classifier_filepath_field.clear()  # На випадок, якщо у полі вже є дані
        file_path = QtWidgets.QFileDialog.getOpenFileName(self, "Оберіть файл",
                                                          filter="*.csv *.data")[0]  # відкрити діалог для обирання
        # директорії i всстановити шлях до файлу як значення змінної
        if file_path:
            self.classifier_filepath_field.setText(file_path)
            self.print_classifier_input(file_path)
        else:
            return

    # Обробка даних
    def process_classifier_data(self):
        if not self.classifier_filepath_field.text() == "":  # Якщо поле шляху не пусте
            path = self.classifier_filepath_field.text()  # Отримання шляху до файлу
            settings = self.get_classification_settings()  # Отримання налаштувань
            classifier.analyze(self, path, settings)  # Відправка даних та налаштувань на обробку
        else:
            self.print_logs('Файл не обрано!')   # Повідомлення про помилку
            return

    # Відкриття файлу та запис даних у поле для вхідних даних
    def print_classifier_input(self, path):
        with open(path, 'r', encoding='utf8') as file:
            self.text_classifier_input_data.setPlainText(file.read())

    # Отримання налаштувань
    def get_classification_settings(self):
        # Роздільник
        if self.rb_classifier_separator_coma.isChecked():
            separator = ','
        elif self.rb_classifier_separator_semicolon.isChecked():
            separator = ';'
        elif self.rb_classifier_separator_space.isChecked():
            separator = ' '

        # Заміна пропущених даних
        if self.rb_classifier_missed_ignore.isChecked():
            missed_values_policy = -1
        elif self.rb_classifier_missed_zeros.isChecked():
            missed_values_policy = 0
        elif self.rb_classifier_missed_calculate.isChecked():
            missed_values_policy = 1

        # Отримання заголовків
        if self.rb_classifier_headers_num.isChecked():
            headers = None
        else:
            headers = 0

        # Отримання індексів рядків
        if self.rb_classifier_rowindex_num.isChecked():
            indexes = None
        elif self.rb_classifier_rowindex_firstcol.isChecked():
            indexes = 0
        else:
            indexes = -1

        # Стопець міток
        if self.rb_classifier_labels_firstcol.isChecked():
            labels = 0
        else:
            labels = -1

        # Частина даних, яка буде використана для передбачення
        try:
            percent_to_predict = float(self.data_to_predict_percent_classifier.text())
        except:
            self.print_logs('Помилка у даних, що введено.Буде використано занченя за змовченням 0,2x')
            percent_to_predict = 0.2

        try:
            neighbors = int(self.heigbors_line.text())
        except:
            self.print_logs('Помилка у даних, що введено.Буде використано занченя за змовченням 0,2x')
            neighbors = 5

        return separator, indexes, headers, labels, missed_values_policy, percent_to_predict, neighbors

    # Друк результатів
    def print_classification_output(self, output_data):
        self.text_classifier_output_data.setPlainText(str(output_data))

    # Регресія =========================================================

    # Отримання шляху до файлу
    def browse_file_regression(self):
        file_path = QtWidgets.QFileDialog.getOpenFileName(self, "Оберіть файл",
                                                          filter="*.csv *.data")[0]  # відкрити діалог для обирання
        # директорії i всстановити шлях до файлу як значення змінної
        if file_path:
            self.regression_filepath_field.clear()  # На випадок, якщо у полі вже є дані
            self.regression_filepath_field.setText(file_path)
            self.print_regression_input(file_path)
        else:
            return

    # Відкриття файлу та запис даних у поле для вхідних даних
    def print_regression_input(self, path):
        with open(path, 'r', encoding='utf8') as file:
            self.text_regression_input_data.setPlainText(file.read())

    # Обробка даних
    def process_regression_data(self):
        if not self.regression_filepath_field.text() == "":  # Якщо поле шляху не пусте
            path = self.regression_filepath_field.text()  # Отримання шляху до файлу
            settings = self.get_regression_settings()  # Отримання налаштувань
            regression.analyze(self, path, settings)  # Відправка даних та налаштувань на обробку
        else:
            self.print_logs('Файл не обрано!')   # Повідомлення про помилку
            return

    # Отримання налаштувань
    def get_regression_settings(self):
        # Роздільник
        if self.rb_regression_separator_coma.isChecked():
            separator = ','
        elif self.rb_regression_separator_semicolon.isChecked():
            separator = ';'
        elif self.rb_regression_separator_space.isChecked():
            separator = ' '

        # Заміна пропущених даних
        if self.rb_regression_missed_ignore.isChecked():
            missed_values_policy = -1
        elif self.rb_regression_missed_zeros.isChecked():
            missed_values_policy = 0
        elif self.rb_regression_missed_calculate.isChecked():
            missed_values_policy = 1

        # Отримання заголовків
        if self.rb_regression_headers_num.isChecked():
            headers = None
        else:
            headers = 0

        # Отримання індексів рядків
        if self.rb_regression_rowindex_num.isChecked():
            indexes = None
        elif self.rb_regression_rowindex_firstcol.isChecked():
            indexes = 0
        else:
            indexes = -1

        # Стопець міток
        if self.rb_regression_labels_firstcol.isChecked():
            labels = 0
        else:
            labels = -1

        # Частина даних, яка буде використана для передбачення
        try:
            percent_to_predict = float(self.data_to_predict_percent_regression.text())
        except:
            self.print_logs('Помилка у даних, що введено.Буде використано занченя за змовченням 0,2')
            percent_to_predict = 0.2

        # Номери стовпців, які буде використано для навчання та тестування
        if not self.target_columns_line_regression.text() =='':
            target_columns = self.convert_string_to_int_list(self.target_columns_line_regression.text(), 4)
        else:
            target_columns = [156, 157, 158, 155]

        return separator, indexes, headers, labels, missed_values_policy, percent_to_predict, target_columns

    # Кластеризація =====================================================
    # Отримання налаштувань
    def get_clustering_settings(self):
        # Початкова ініціалізація змінних
        n_clusters = None
        data = None
        epsilon = None
        minpts = None
        # Отримання обраних чекбоксів
        kmeans = self.checkBox_kmeans.isChecked()
        tsne = self.checkBox_tsne.isChecked()
        dbscan = self.checkBox_dbscan.isChecked()

        # Отримання введеної кількості кластерів
        if not self.clusters.text() == '':
            n_clusters = int(self.clusters.text())

        # Отримання введених даних для передбачення
        if not self.data_to_predict.text() == '':
            data = self.convert_string_to_float_list(self.data_to_predict.text(), 4)

        # Повідомлення осей для відображення графіків
        try:
            axes = self.convert_string_to_int_list(self.clustering_axes.text(), 2)
        except:
            self.print_logs('Помилка при зчитуванні даних. Буде використано значення за змовченням [0, 1]')
            axes = [0, 1]

        if not self.dbscan_eps.text() == '':
            epsilon = float(self.dbscan_eps.text())
        if not self.dbscan_minpts.text() == '':
            minpts = float(self.dbscan_minpts.text())

        return n_clusters, data, kmeans, tsne, dbscan, axes, epsilon, minpts

    # Обробка даних
    def start_clustering(self):
        self.text_clustering_output_data.clear()
        settings = self.get_clustering_settings()
        clustering.start(self, settings)

    # Відкриття файлу та запис даних у поле для вхідних даних
    def print_clustering_input(self, input_data):
        self.text_clustering_input_data.setPlainText(input_data)

    # Друк результатів
    def print_clustering_output(self, output_data):
        self.text_clustering_output_data.setPlainText(str(output_data))

    # Ввікнення полів для налаштуавння методу к-середніх
    def enable_kmeans_settings(self):
        if self.checkBox_kmeans.isChecked():
            self.clusters.setDisabled(False)
            self.data_to_predict.setDisabled(False)
        else:
            self.clusters.setDisabled(True)
            self.data_to_predict.setDisabled(True)

    def enable_dbscan_settings(self):
        if self.checkBox_dbscan.isChecked():
            self.dbscan_eps.setDisabled(False)
            self.dbscan_minpts.setDisabled(False)
        else:
            self.dbscan_eps.setDisabled(True)
            self.dbscan_minpts.setDisabled(True)

    # Друк логів
    def print_logs(self, message):
        self.logs_label.clear()
        self.logs_label.setText(message)

    # Конвертування рядка у список типу float заданого розміру
    def convert_string_to_float_list(self, string_numbers, amount):
        string_numbers_list = string_numbers.split()
        if not len(string_numbers_list) == amount:
            self.print_logs('Wrong data!')
            return
        else:
            return list(map(float, string_numbers_list))

    # Конвертування рядка у список типу int заданого розміру
    def convert_string_to_int_list(self, string_numbers, amount):
        string_numbers_list = string_numbers.split()
        if not len(string_numbers_list) == amount:
            self.print_logs('Wrong data!')
            return
        else:
            return list(map(int, string_numbers_list))


# Метод, що запускає пограму
def main():
    app = QtWidgets.QApplication(sys.argv)  # Новий екземпляр QApplication
    window = MachineLearningApp()  # Створення об'єкту класу ExampleApp
    window.show()  # Відображення вікна
    app.exec_()  # запуск додатку


if __name__ == '__main__':  # Для запуску напряму (без імпорту у іншу програму)
    main()
