# from sklearn.metrics import recall_score
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor as xgbmodel
from matplotlib import pyplot
import pandas as pd
import numpy as np
import time
np.random.seed(0)


def feature_importances_xgboost(dataset, df_cols, target, train_per=0.6, valid_per=0.0):
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
    model = xgbmodel(objective='reg:squarederror')
    time_start = time.time()
    # model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)])
    # model.fit(X_train, y_train, eval_metric="mae", eval_set=[(X_valid, y_valid)], verbose=False)
    model.fit(X_train, y_train, eval_metric="mae", verbose=False)

    print(model.feature_importances_)
    # plot
    pyplot.figure(figsize=(16, 7))
    pyplot.barh(list(df_cols[:-1]), model.feature_importances_)
    pyplot.xticks(rotation=90)
    pyplot.savefig("fi_{}.png".format(target))
    pyplot.close()

    all_results = model.predict(X)
    mae_all = mean_absolute_error(Y, all_results)
    r2_all = r2_score(Y, all_results)

    train_results = model.predict(X_train)
    mae_train = mean_absolute_error(y_train, train_results)
    r2_train = r2_score(y_train, train_results)

    test_results = model.predict(X_test)
    mae_test = mean_absolute_error(y_test, test_results)
    r2_test = r2_score(y_test, test_results)

    # print("MAE_all_data: %.3f" % (mae_all))
    print("R2_all_data: %.3f" % (r2_all))
    # print("MAE_train: %.3f" % (mae_train))
    print("R2_train: %.3f" % (r2_train))
    # print("MAE_test: %.3f" % (mae_test))
    print("R2_test: %.3f" % (r2_test))
    print("Time training: ", time.time() - time_start)


if __name__ == "__main__":
    dataset = pd.read_csv('data/hoat_chat.csv')
    dataset = dataset.drop(columns=["Date", "Site", "Sample"])
    target = "THMFP"
    target = "HAAFP"
    dataset = dataset.dropna()
    if target == "THMFP":
        # cols[-2], cols[-1] = cols[-1], cols[-2]
        # dataset = dataset.reindex(columns=cols)
        dataset = dataset.drop(columns=["HAAFP"])
    else:
        dataset = dataset.drop(columns=["THMFP"])
    cols = list(dataset.columns)
    print(cols)
    feature_importances_xgboost(dataset, cols, target)