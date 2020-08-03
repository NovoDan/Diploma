from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# Завантаження набору даних
iris_df = datasets.load_iris()


def plot_origin_dataset(axes, user_xaxis, user_yaxis):
    """
    Друкує початкові дані та готує графік розподілу цих даних
    """

    # Розділення набору даних на осі
    x_axis = iris_df.data[:, user_xaxis]
    y_axis = iris_df.data[:, user_yaxis]

    # Побудова даних гарфіку
    axes[0][0].scatter(x_axis, y_axis, c=iris_df.target)
    axes[0][0].set_xlabel('Початкові дані')


def k_means_method(axes, clusters, data_to_predict, user_xaxis, user_yaxis):
    """
    Навчання за методом к-середніх та підготовка результатів для друку на виводу графіку
    """

    predicted_label_result = ''
    # Опис моделі
    model = KMeans(n_clusters=clusters)

    # Моделювання
    model.fit(iris_df.data)

    # Передбачення на одничному прикладі
    if data_to_predict:
        predicted_label = model.predict([data_to_predict])
        predicted_label_result = "Передбачена мітка даних: {}\n".format(predicted_label)

    # Передбачення на всьому наборі даних
    all_predictions = model.predict(iris_df.data)

    # Отримання результатів передбачення
    all_predictions_result = 'Передбачені міткі (Метод К-cередніх):\n {}'.format(all_predictions)

    # Розділення набору даних
    x_axis = iris_df.data[:, user_xaxis]
    y_axis = iris_df.data[:, user_yaxis]

    # Побудова масивів для граіиків
    axes[0][1].scatter(x_axis, y_axis, c=all_predictions)
    axes[0][1].set_xlabel('Метод К-середніх')

    result = predicted_label_result + '\n' + all_predictions_result

    return result


def tSNE_method(axes, user_xaxis, user_yaxis, clusters):
    """
    Навчання за методом t-SNE та підготовка результатів для друку на виводу графіку
    """
    # Визначення моделі та швидкості навчання
    model = TSNE()

    # навчання моделі
    transformed = model.fit_transform(iris_df.data)

    model = KMeans(n_clusters=clusters)
    model.fit(transformed)

    # Передбачення на всьому наборі даних
    all_predictions = model.predict(transformed)

    # Розділення набору даних
    x_axis = transformed[:, user_xaxis]
    y_axis = transformed[:, user_yaxis]

    axes[1][0].scatter(x_axis, y_axis, c=all_predictions)
    axes[1][0].set_xlabel('Метод К-середніх зі зменш. розм.')

    return 'Передбачені міткі (Метод К-cередніх зі зменш. розм.):\n {}'.format(all_predictions)


def dbscan_method(axes, epsilon, samples):
    """
        Навчання за методом DBSCAN та підготовка результатів для друку на виводу графіку
    """
    # Визначення моделі
    dbscan = DBSCAN(eps=epsilon, min_samples=samples)

    # Навчання моделі
    dbscan.fit(iris_df.data)

    # Зменшення розмірності за допомогою методу головних компонент
    pca = PCA(n_components=2).fit(iris_df.data)
    pca_2d = pca.transform(iris_df.data)

    # Побудова результатів з виділенням шумів
    noise = None
    axes[1][1].scatter(pca_2d[:, 0], pca_2d[:, 1], c=dbscan.labels_)
    for i in range(0, pca_2d.shape[0]):
        if dbscan.labels_[i] == -1:
            noise = axes[1][1].scatter(pca_2d[i, 0], pca_2d[i, 1], c='r')

    if noise:
        axes[1][1].legend([noise], ['Шум'])
    axes[1][1].set_xlabel('Метод DBSCAN')

    return 'Передбачені міткі (Метод DBSCAN):\n {}'.format(dbscan.labels_)


def start(app, settings):
    result = ''

    n_clusters = 3
    data_to_predict = None
    user_xaxis = 0
    user_yaxis = 1
    epsilon = 0.5
    samples = 5

    if settings[5]:
        user_axes = settings[5]
        user_xaxis = user_axes[0]
        user_yaxis = user_axes[1]

    # Визначення максимальної кількості графіків
    fig, axes = plt.subplots(nrows=2, ncols=2)
    app.print_logs('Початок кластеризації...')
    # Вивід вхідних даних
    app.print_clustering_input('Ознаки: {} \nІмена міток: {} \nМітки: {} \nДані: {}'.format(iris_df.feature_names,
                                                                                            iris_df.target_names,
                                                                                            iris_df.target,
                                                                                            iris_df.data))
    # Створення графіку початкових даних
    plot_origin_dataset(axes, user_xaxis, user_yaxis)
    # Обір налаштувань
    if settings[0]:
        n_clusters = settings[0]
    if settings[1]:
        data_to_predict = settings[1]
    if settings[6]:
        epsilon = settings[6]
    if settings[7]:
        samples = settings[7]
    if settings[2]:
        result += '\n' + k_means_method(axes, n_clusters, data_to_predict, user_xaxis, user_yaxis)
    if settings[3]:
        result += '\n' + tSNE_method(axes, user_xaxis, user_yaxis, n_clusters)
    if settings[4]:
        result += '\n' + dbscan_method(axes, epsilon, samples)
    # Друк результатів
    app.print_clustering_output(result)
    plt.show()

    plt.close()
    return
