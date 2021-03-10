from sklearn.metrics import recall_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import time
np.random.seed(0)


def feature_importances_xgboost(dataset, train_per=0.6, valid_per=0.2):
    dataset = dataset.to_numpy()
    X = dataset[:, 0:-1]
    Y = dataset[:, -1]
    # split data into train and test sets
    train_size = int(len(dataset)*train_per)
    valid_size = int(len(dataset)*valid_per)
    X_train = X[0:train_size]
    y_train = Y[0:train_size]
    X_valid = X[train_size:train_size+valid_size]
    y_valid = Y[train_size:train_size+valid_size]
    X_test = X[train_size+valid_size:]
    y_test = Y[train_size+valid_size:]
    model = XGBClassifier()
    time_start = time.time()
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)])
    test_results = model.predict(X_test)
    recall = recall_score(y_test, test_results)
    print("Recall: %.3f" % (recall))
    print("Time training: ", time.time() - time_start)


if __name__ == "__main__":
    dataset = pd.read_csv('data/pima_indians_diabetes.csv')
    feature_importances_xgboost(dataset)
