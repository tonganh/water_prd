# from sklearn.metrics import recall_score
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor as xgbmodel
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
    # model = xgbmodel()
    model = model = xgbmodel(objective ='reg:squarederror', max_depth=8, n_estimators=1000, min_child_weight=300, colsample_bytree=0.8, 
    subsample=0.8, eta=0.3, seed=2)

    
    time_start = time.time()
    # model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)])
    model.fit(X_train, y_train, eval_metric="mae", eval_set=[(X_valid, y_valid)], verbose=False)
    test_results = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_results)
    print("MAE: %.3f" % (mae))
    print("Time training: ", time.time() - time_start)


if __name__ == "__main__":
    dataset = pd.read_csv('data/credit_card.csv')
    feature_importances_xgboost(dataset)
