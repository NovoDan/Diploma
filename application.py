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
        self.btn_browse_classification.clicked.connect(self.browse_file_classifier)  # Прив'язати функцію browse_file до кнопки
        self.btn_browse_regression.clicked.connect(self.browse_file_regression)
        # Завантаення даних
        self.btn_download_classifier_data.clicked.connect(self.process_classifier_data)
        self.btn_download_regression_data.clicked.connect(self.process_regression_data)
        self.btn_start_clustering.clicked.connect(self.start_clustering)
        self.checkBox_kmeans.clicked.connect(self.enable_kmeans_settings)

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

    def process_classifier_data(self):
        if not self.classifier_filepath_field.text() == "":
            path = self.classifier_filepath_field.text()
            settings = self.get_classification_settings()
            classifier.analyze(self, path, settings)
        else:
            self.print_logs('Файл не обрано!')
            return

    def print_classifier_input(self, path):
        with open(path, 'r', encoding='utf8') as file:
            self.text_classifier_input_data.setPlainText(file.read())

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

        return separator, indexes, headers, labels, missed_values_policy, percent_to_predict

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

    def print_regression_input(self, path):
        with open(path, 'r', encoding='utf8') as file:
            self.text_regression_input_data.setPlainText(file.read())

    def process_regression_data(self):
        if not self.regression_filepath_field.text() == "":
            path = self.regression_filepath_field.text()
            settings = self.get_regression_settings()
            regression.analyze(self, path, settings)
        else:
            self.print_logs('Файл не обрано!')
            return

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
        try:
            target_columns = self.convert_string_to_float_list(self.target_column_line_regression.text())
        except:
            self.print_logs('Помилка у даних, що введено.Буде використано занченя за змовченням [156, 157, 158, 155]')
            target_columns = [156, 157, 158, 155]

        return separator, indexes, headers, labels, missed_values_policy, percent_to_predict, target_columns

    # Кластеризація =====================================================

    def get_clustering_settings(self):
        n_clusters = None
        data = None
        kmeans = self.checkBox_kmeans.isChecked()
        tsne = self.checkBox_tsne.isChecked()
        dbscan = self.checkBox_dbscan.isChecked()

        if not self.clusters.text() == '':
            n_clusters = self.convert_string_to_float_list(self.clusters.text())
        if not self.data_to_predict.text() == '':
            data = self.data_to_predict.text()

        return n_clusters, data, kmeans, tsne, dbscan

    def start_clustering(self):
        self.text_clustering_output_data.clear()
        settings = self.get_clustering_settings()
        clustering.start(self, settings)

    def print_clustering_input(self, input_data):
        self.text_clustering_input_data.setPlainText(input_data)

    def print_clustering_output(self, output_data):
        self.text_clustering_output_data.setPlainText(str(output_data))

    def enable_kmeans_settings(self):
        if self.checkBox_kmeans.isChecked():
            self.clusters.setDisabled(False)
            self.data_to_predict.setDisabled(False)
        else:
            self.clusters.setDisabled(True)
            self.data_to_predict.setDisabled(True)
        # return

    # Друк логів
    def print_logs(self, message):
        self.logs_label.clear()
        self.logs_label.setText(message)

    def convert_string_to_float_list(self, string_numbers):
        string_numbers_list = string_numbers.split()
        if not len(string_numbers_list) == 4:
            self.print_logs('Wrong data to predict!')
            return
        else:
            return map(float, string_numbers_list)


def main():
    app = QtWidgets.QApplication(sys.argv)  # Новий екземпляр QApplication
    window = MachineLearningApp()  # Створення об'єкту класу ExampleApp
    window.show()  # Відображення вікна
    app.exec_()  # запуск додатку


if __name__ == '__main__':  # Для запуску напряму (без імпорту у іншій програмі)
    main()
