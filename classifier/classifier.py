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

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
# import matplotlib
# matplotlib.use('Agg')

from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt

# Import some classifiers to test
from sklearn.svm import LinearSVC, NuSVC
from sklearn.ensemble import AdaBoostClassifier

# We will calculate the P-R curve for each classifier
from sklearn.metrics import precision_recall_curve, f1_score

try:
    import seaborn
except ImportError:
    pass


# =====================================================================


def download_data(path):
    """
    Downloads the data for this script into a pandas DataFrame.
    """

    frame = read_table(
        path,

        # Specify the file encoding
        encoding='utf-8',

        # Specify the separator in the data
        sep=',',

        # Ignore spaces after the separator
        skipinitialspace=True,

        # Generate row labels from each row number
        index_col=None,

        # Generate column headers row from each column number
        header=None,

    )

    # Return a subset of the columns
    # return frame[['col1', 'col4', ...]]

    # Return the entire frame
    return frame


# =====================================================================


def get_features_and_labels(frame):
    """
    Transforms and scales the input data and returns numpy arrays for
    training and testing inputs and targets.
    """

    # Convert values to floats
    arr = np.array(frame, dtype=np.float)

    # Use the last column as the target value
    X, y = arr[:, :-1], arr[:, -1]
    # To use the first column instead, change the index value
    # X, y = arr[:, 1:], arr[:, 0]

    # Use 80% of the data for training; test against the rest
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # sklearn.pipeline.make_pipeline could also be used to chain 
    # processing and classification into a black box, but here we do
    # them separately.

    # If values are missing we could impute them from the training data
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)

    # Normalize the attribute values to mean=0 and variance=1
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # To scale to a specified range, use MinMaxScaler
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit the scaler based on the training data, then apply the same
    # scaling to both training and test sets.
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Return the training and test sets
    return X_train, X_test, y_train, y_test


# =====================================================================


def evaluate_classifier(X_train, X_test, y_train, y_test):
    """
    Run multiple times with different classifiers to get an idea of the
    relative performance of each configuration.

    Returns a sequence of tuples containing:
        (title, precision, recall)
    for each learner.
    """

    # Here we create classifiers with default parameters. These need
    # to be adjusted to obtain optimal performance on your data set.

    # Test the linear support vector classifier
    classifier = LinearSVC(C=1)
    # Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'Linear SVC (F1 score={:.3f})'.format(score), precision, recall

    # Test the Nu support vector classifier
    classifier = NuSVC(kernel='rbf', nu=0.5, gamma=1e-3)
    # Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'NuSVC (F1 score={:.3f})'.format(score), precision, recall

    # Test the Ada boost classifier
    classifier = AdaBoostClassifier(n_estimators=50, learning_rate=1.0, algorithm='SAMME.R')
    # Fit the classifier
    classifier.fit(X_train, y_train)
    score = f1_score(y_test, classifier.predict(X_test))
    # Generate the P-R curve
    y_prob = classifier.decision_function(X_test)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    # Include the score in the title
    yield 'Ada Boost (F1 score={:.3f})'.format(score), precision, recall


# =====================================================================


def plot(results, URL):
    """
    Create a plot comparing multiple learners.

    `results` is a list of tuples containing:
        (title, precision, recall)

    All the elements in results will be plotted.
    """

    # Plot the precision-recall curves

    fig = plt.figure(figsize=(6, 6))
    fig.canvas.set_window_title('Classifying data from ' + URL)

    for label, precision, recall in results:
        plt.plot(recall, precision, label=label)

    plt.title('Precision-Recall Curves')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend(loc='lower left')

    # Let matplotlib improve the layout
    plt.tight_layout()

    # ==================================
    # Display the plot in interactive UI
    plt.show()

    # To save the plot to an image file, use savefig()
    # plt.savefig('plot.png')

    # Open the image file with the default image viewer
    # import subprocess
    # subprocess.Popen('plot.png', shell=True)

    # Closing the figure allows matplotlib to release the memory used.
    plt.close()


# =====================================================================

def analyze(app, path):
    app.print_logs("Завантаження даних з {}...".format(path))
    frame = download_data(path)

    # Process data into feature and label arrays
    app.print_logs("Обробка {} зразків з {} атрибутами...".format(len(frame.index), len(frame.columns)))
    X_train, X_test, y_train, y_test = get_features_and_labels(frame)

    # Evaluate multiple classifiers on the data
    app.print_logs("Оцінювання класифікаторів...")
    results = list(evaluate_classifier(X_train, X_test, y_train, y_test))

    # Display the results
    app.print_logs("Друк результатів...")
    plot(results, path)

# if __name__ == '__main__':
#     # Download the data set from URL
#     print("Downloading data from {}".format(URL))
#     frame = download_data()
#
#     # Process data into feature and label arrays
#     print("Processing {} samples with {} attributes".format(len(frame.index), len(frame.columns)))
#     X_train, X_test, y_train, y_test = get_features_and_labels(frame)
#
#     # Evaluate multiple classifiers on the data
#     print("Evaluating classifiers")
#     results = list(evaluate_classifier(X_train, X_test, y_train, y_test))
#
#     # Display the results
#     print("Plotting the results")
#     plot(results)