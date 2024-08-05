import os
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score


##########################################################
#             MNIST784 Image Classification              #
##########################################################

def get_mnist784_dataset():
    data_dir = os.path.join(os.getcwd(), "datasets")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    mnist_dir =  os.path.join(os.getcwd(), "datasets", "mnist_784")
    if not os.path.exists(mnist_dir):
        os.mkdir(mnist_dir)

    mnist = fetch_openml('mnist_784', as_frame=False)
    data_df = pd.DataFrame(mnist.data, columns=mnist.feature_names)
    target_df = pd.DataFrame(mnist.target, columns=mnist.target_names)
    data_path = os.path.join(os.getcwd(), "datasets", "mnist_784", "mnist_784_data.csv")
    target_path = os.path.join(os.getcwd(), "datasets", "mnist_784", "mnist_784_target.csv")
    data_df.to_csv(data_path)
    target_df.to_csv(target_path)
    X, y = mnist.data, mnist.target
    return X, y

def mnist784_df_from_csv():
    data_path = os.path.join(os.getcwd(), "datasets", "mnist_784", "mnist_784_data.csv")
    target_path = os.path.join(os.getcwd(), "datasets", "mnist_784", "mnist_784_target.csv")
    data_df = pd.read_csv(data_path)
    target_df = pd.read_csv(target_path)
    X_df, y_df = data_df, target_df
    X_df.drop(columns="Unnamed: 0", inplace=True)
    y_df.drop(columns="Unnamed: 0", inplace=True)
    return X_df, y_df

def train_predict_mnist784():
    X_df, y_df = mnist784_df_from_csv()
    X = X_df.values
    y = y_df.values
    some_digit = X[0]
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X_train, y_train_5)
    sgd_clf.predict([some_digit])
    scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
    return scores



