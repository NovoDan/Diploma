from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Завантаження набору даних
iris_df = datasets.load_iris()


def plot_origin_dataset(axes):

    # Розділення набору даних
    x_axis = iris_df.data[:, 0]  # Sepal Length
    y_axis = iris_df.data[:, 1]  # Sepal Width

    # Побудова даних гарфіку
    axes[0][0].scatter(x_axis, y_axis, c=iris_df.target)
    axes[0][0].set_xlabel('Початкові дані')


def k_means_method(axes, data_to_predict):
    predicted_label_result = ''
    # Опис моделі
    model = KMeans(n_clusters=3)

    # Моделювання
    model.fit(iris_df.data)

    # Передбачення на одничному прикладі
    if data_to_predict:
        predicted_label = model.predict([[7.0, 3.2, 4.7, 1.4]])
        predicted_label_result = "Predicted: {}".format(predicted_label)

    # Передбачення на всьому наборі даних
    all_predictions = model.predict(iris_df.data)

    # Отримання результатів передбачення
    all_predictions_result = 'Передбачені міткі (Метод К-ередніх):\n {}'.format(model.predict(iris_df.data))

    # Розділення набору даних
    x_axis = iris_df.data[:, 0]  # Sepal Length
    y_axis = iris_df.data[:, 1]  # Sepal Width

    # Побудова масивів для граіиків
    axes[0][1].scatter(x_axis, y_axis, c=all_predictions)
    axes[0][1].set_xlabel('Метод К-середніх')

    result = predicted_label_result + '\n' + all_predictions_result

    return result


def tSNE_method(axes):
    # Визначення моделі на швидкості навчання
    model = TSNE(learning_rate=100)

    # навчання моделі
    transformed = model.fit_transform(iris_df.data)

    # Трансформування результату у двовимірний
    x_axis = transformed[:, 0]
    y_axis = transformed[:, 1]

    axes[1][0].scatter(x_axis, y_axis, c=iris_df.target)
    axes[1][0].set_xlabel('Метод t-SNE')


def dbscan_method(axes):
    # Визначення моделі
    dbscan = DBSCAN()

    # Навчання моделі
    dbscan.fit(iris_df.data)

    # Зменшення розмірності за допомогою методу головних компонент
    pca = PCA(n_components=2).fit(iris_df.data)
    pca_2d = pca.transform(iris_df.data)

    # Побудова результатів у відповідності до трьох класів
    for i in range(0, pca_2d.shape[0]):
        if dbscan.labels_[i] == 0:
            c1 = axes[1][1].scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
        elif dbscan.labels_[i] == 1:
            c2 = axes[1][1].scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
        elif dbscan.labels_[i] == -1:
            c3 = axes[1][1].scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='*')

    axes[1][1].legend([c1, c2, c3], ['Кластер 1', 'Кластер 2', 'Шум'])
    axes[1][1].set_xlabel('Метод t-SNE')


def start(app, settings):
    data_to_predict = None
    # Визначення максимальної кількості графіків
    fig, axes = plt.subplots(nrows=2, ncols=2)
    app.print_logs('Початок кластеризації...')
    # Вивід вхідних даних
    app.print_clustering_input('Ознаки: {} \nІмена міток: {} \nМітки: {} \nДані: {}'.format(iris_df.feature_names,
                                                                                            iris_df.target_names,
                                                                                            iris_df.target,
                                                                                            iris_df.data))
    # Створення графіку початкових даних
    plot_origin_dataset(axes)
    # Обір налаштувань
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
    # Друк результатів
    plt.show()

    plt.close()
    return
