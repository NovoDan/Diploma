from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Загружаем набор данных
iris_df = datasets.load_iris()


def plot_origin_dataset(axes):

    # Разделение набора данных
    x_axis = iris_df.data[:, 0]  # Sepal Length
    y_axis = iris_df.data[:, 1]  # Sepal Width

    # Построение
    axes[0][0].scatter(x_axis, y_axis, c=iris_df.target)
    axes[0][0].set_xlabel('Origin')


def k_means_method(axes, data_to_predict):
    predicted_label_result = ''
    # Описываем модель
    model = KMeans(n_clusters=3)

    # Проводим моделирование
    model.fit(iris_df.data)

    # Предсказание на единичном примере
    if data_to_predict:
        predicted_label = model.predict([[7.0, 3.2, 4.7, 1.4]])
        predicted_label_result = "Predicted: {}".format(predicted_label)

    # Предсказание на всем наборе данных
    all_predictions = model.predict(iris_df.data)

    # Выводим предсказания
    all_predictions_result = 'Передбачені міткі (Метод К-ередніх):\n {}'.format(model.predict(iris_df.data))

    # Разделение набора данных
    x_axis = iris_df.data[:, 0]  # Sepal Length
    y_axis = iris_df.data[:, 1]  # Sepal Width

    # Построение
    axes[0][1].scatter(x_axis, y_axis, c=all_predictions)
    axes[0][1].set_xlabel('K-Means')

    result = predicted_label_result + '\n' + all_predictions_result

    return result


def tSNE_method(axes):
    # Определяем модель и скорость обучения
    model = TSNE(learning_rate=100)

    # Обучаем модель
    transformed = model.fit_transform(iris_df.data)

    # Представляем результат в двумерных координатах
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]

    # plt.scatter(x_axis, y_axis, c=iris_df.target)
    axes[1][0].scatter(x_axis, y_axis, c=iris_df.target)
    axes[1][0].set_xlabel('t-SNE')


def dbscan_method(axes):
    # Определяем модель
    dbscan = DBSCAN()

    # Обучаем
    dbscan.fit(iris_df.data)

    # Уменьшаем размерность при помощи метода главных компонент
    pca = PCA(n_components=2).fit(iris_df.data)
    pca_2d = pca.transform(iris_df.data)

    # Строим в соответствии с тремя классами
    for i in range(0, pca_2d.shape[0]):
        if dbscan.labels_[i] == 0:
            c1 = axes[1][1].scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
        elif dbscan.labels_[i] == 1:
            c2 = axes[1][1].scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        elif dbscan.labels_[i] == -1:
            c3 = axes[1][1].scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

    axes[1][1].legend([c1, c2, c3], ['Кластер 1', 'Кластер 2', 'Шум'])
    axes[1][1].set_xlabel('t-SNE')


def start(app, settings):
    data_to_predict = None
    fig, axes = plt.subplots(nrows=2, ncols=2)
    app.print_logs('Start clustering...')
    app.print_clustering_input('Ознаки: {} \nІмена міток: {} \nМітки: {} \nДані: {}'.format(iris_df.feature_names,
                                                                                            iris_df.target_names,
                                                                                            iris_df.target,
                                                                                            iris_df.data))
    plot_origin_dataset(axes)
    if settings[0]:
        n_clusters = settings[0]
    if settings[1]:
        data_to_predict = settings[1]
    if settings[2]:
        result = k_means_method(axes, data_to_predict)
        app.print_clustering_output(result)
    if settings[3]:
        tSNE_method(axes)
    if settings[4]:
        dbscan_method(axes)
    plt.show()

    plt.close()
    return
