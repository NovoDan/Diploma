# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pydesign.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(917, 795)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.classification_tab = QtWidgets.QWidget()
        self.classification_tab.setObjectName("classification_tab")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.classification_tab)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.classifier_filepath_field = QtWidgets.QLineEdit(self.classification_tab)
        self.classifier_filepath_field.setObjectName("classifier_filepath_field")
        self.gridLayout_3.addWidget(self.classifier_filepath_field, 0, 0, 1, 1)
        self.btn_browse_classification = QtWidgets.QPushButton(self.classification_tab)
        self.btn_browse_classification.setObjectName("btn_browse_classification")
        self.gridLayout_3.addWidget(self.btn_browse_classification, 0, 1, 1, 1)
        self.text_classifier_input_data = QtWidgets.QTextEdit(self.classification_tab)
        self.text_classifier_input_data.setObjectName("text_classifier_input_data")
        self.gridLayout_3.addWidget(self.text_classifier_input_data, 3, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.classification_tab)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 2, 0, 1, 1)
        self.btn_download_classifier_data = QtWidgets.QPushButton(self.classification_tab)
        self.btn_download_classifier_data.setObjectName("btn_download_classifier_data")
        self.gridLayout_3.addWidget(self.btn_download_classifier_data, 1, 0, 1, 1)
        self.tabWidget.addTab(self.classification_tab, "")
        self.regression_tab = QtWidgets.QWidget()
        self.regression_tab.setObjectName("regression_tab")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.regression_tab)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.text_regression_input_data = QtWidgets.QTextEdit(self.regression_tab)
        self.text_regression_input_data.setObjectName("text_regression_input_data")
        self.gridLayout_2.addWidget(self.text_regression_input_data, 3, 0, 1, 1)
        self.btn_browse_regression = QtWidgets.QPushButton(self.regression_tab)
        self.btn_browse_regression.setObjectName("btn_browse_regression")
        self.gridLayout_2.addWidget(self.btn_browse_regression, 0, 1, 1, 1)
        self.regression_filepath_field = QtWidgets.QLineEdit(self.regression_tab)
        self.regression_filepath_field.setObjectName("regression_filepath_field")
        self.gridLayout_2.addWidget(self.regression_filepath_field, 0, 0, 1, 1)
        self.btn_download_regression_data = QtWidgets.QPushButton(self.regression_tab)
        self.btn_download_regression_data.setObjectName("btn_download_regression_data")
        self.gridLayout_2.addWidget(self.btn_download_regression_data, 1, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.regression_tab)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 2, 0, 1, 1)
        self.tabWidget.addTab(self.regression_tab, "")
        self.clasterization_tab = QtWidgets.QWidget()
        self.clasterization_tab.setObjectName("clasterization_tab")
        self.gb_clustering_settings = QtWidgets.QGroupBox(self.clasterization_tab)
        self.gb_clustering_settings.setGeometry(QtCore.QRect(10, 20, 881, 121))
        self.gb_clustering_settings.setObjectName("gb_clustering_settings")
        self.label = QtWidgets.QLabel(self.gb_clustering_settings)
        self.label.setGeometry(QtCore.QRect(200, 30, 101, 16))
        self.label.setObjectName("label")
        self.clusters = QtWidgets.QLineEdit(self.gb_clustering_settings)
        self.clusters.setEnabled(False)
        self.clusters.setGeometry(QtCore.QRect(310, 30, 31, 20))
        self.clusters.setText("")
        self.clusters.setObjectName("clusters")
        self.label_2 = QtWidgets.QLabel(self.gb_clustering_settings)
        self.label_2.setGeometry(QtCore.QRect(200, 70, 81, 31))
        self.label_2.setObjectName("label_2")
        self.checkBox_kmeans = QtWidgets.QCheckBox(self.gb_clustering_settings)
        self.checkBox_kmeans.setGeometry(QtCore.QRect(20, 20, 131, 17))
        self.checkBox_kmeans.setObjectName("checkBox_kmeans")
        self.checkBox_tsne = QtWidgets.QCheckBox(self.gb_clustering_settings)
        self.checkBox_tsne.setGeometry(QtCore.QRect(20, 50, 101, 17))
        self.checkBox_tsne.setObjectName("checkBox_tsne")
        self.checkBox_dbscan = QtWidgets.QCheckBox(self.gb_clustering_settings)
        self.checkBox_dbscan.setGeometry(QtCore.QRect(20, 80, 111, 17))
        self.checkBox_dbscan.setObjectName("checkBox_dbscan")
        self.btn_start_clustering = QtWidgets.QPushButton(self.gb_clustering_settings)
        self.btn_start_clustering.setGeometry(QtCore.QRect(650, 30, 131, 61))
        self.btn_start_clustering.setObjectName("btn_start_clustering")
        self.data_to_predict = QtWidgets.QLineEdit(self.gb_clustering_settings)
        self.data_to_predict.setEnabled(False)
        self.data_to_predict.setGeometry(QtCore.QRect(290, 80, 113, 20))
        self.data_to_predict.setText("")
        self.data_to_predict.setObjectName("data_to_predict")
        self.groupBox = QtWidgets.QGroupBox(self.clasterization_tab)
        self.groupBox.setGeometry(QtCore.QRect(10, 160, 881, 521))
        self.groupBox.setObjectName("groupBox")
        self.text_clustering_input_data = QtWidgets.QTextEdit(self.groupBox)
        self.text_clustering_input_data.setGeometry(QtCore.QRect(10, 60, 431, 441))
        self.text_clustering_input_data.setObjectName("text_clustering_input_data")
        self.text_clustering_output_data = QtWidgets.QTextEdit(self.groupBox)
        self.text_clustering_output_data.setGeometry(QtCore.QRect(450, 60, 421, 441))
        self.text_clustering_output_data.setObjectName("text_clustering_output_data")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(20, 40, 71, 16))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(450, 40, 101, 16))
        self.label_4.setObjectName("label_4")
        self.tabWidget.addTab(self.clasterization_tab, "")
        self.verticalLayout.addWidget(self.tabWidget)
        self.logs_label = QtWidgets.QLabel(self.centralwidget)
        self.logs_label.setText("")
        self.logs_label.setObjectName("logs_label")
        self.verticalLayout.addWidget(self.logs_label)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Машинне навчання"))
        self.btn_browse_classification.setText(_translate("MainWindow", "Обрати файл"))
        self.label_6.setText(_translate("MainWindow", "Вхідні дані"))
        self.btn_download_classifier_data.setText(_translate("MainWindow", "Завантажити дані"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.classification_tab), _translate("MainWindow", "Класифікація"))
        self.btn_browse_regression.setText(_translate("MainWindow", "Обрати файл"))
        self.btn_download_regression_data.setText(_translate("MainWindow", "Завантажити дані"))
        self.label_5.setText(_translate("MainWindow", "Вхідні дані"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.regression_tab), _translate("MainWindow", "Регресія"))
        self.gb_clustering_settings.setTitle(_translate("MainWindow", "НАЛАШТУВАННЯ"))
        self.label.setText(_translate("MainWindow", "Кількість кластерів"))
        self.clusters.setPlaceholderText(_translate("MainWindow", "3"))
        self.label_2.setText(_translate("MainWindow", "Дані для \n"
"передбачення"))
        self.checkBox_kmeans.setText(_translate("MainWindow", "Метод К-середніх"))
        self.checkBox_tsne.setText(_translate("MainWindow", "Метод t-SNE"))
        self.checkBox_dbscan.setText(_translate("MainWindow", "Метод DBSCAN"))
        self.btn_start_clustering.setText(_translate("MainWindow", "Старт"))
        self.data_to_predict.setPlaceholderText(_translate("MainWindow", "n1 n2 n3 n4"))
        self.groupBox.setTitle(_translate("MainWindow", "РЕЗУЛЬТАТИ"))
        self.label_3.setText(_translate("MainWindow", "Вхідні дані"))
        self.label_4.setText(_translate("MainWindow", "Передбачені дані"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.clasterization_tab), _translate("MainWindow", "Кластерізація"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
