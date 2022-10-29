from pprint import pprint
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from common import seed_everything, split_data
import warnings
import numpy as np
import importlib
from random import random, randrange, shuffle
warnings.filterwarnings(
    action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')


def append_result(model, mae_train, r2_train, mae_test, r2_test):
    results[model]["mae_train"].append(mae_train)
    results[model]["r2_train"].append(r2_train)
    results[model]["mae_test"].append(mae_test)
    results[model]["r2_test"].append(r2_test)


def get_engine(model):
    package = importlib.import_module('common_kFold_train')
    model_engine = getattr(package, "run_{}".format(model))
    return model_engine


def train_test_split_custom(dataset, split):
    train_dataset = list()
    train_size = split * len(dataset)
    test_dataset = list(dataset)
    while len(train_dataset) < train_size:
        index = randrange(len(test_dataset))
        train_dataset.append(test_dataset.pop(index))
    return train_dataset, test_dataset


if __name__ == "__main__":
    seed_everything(50)
    # list_features = {"water": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "Chl-a"],
    #                   "water_optics": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "C1", "C2", "C3", "Chl-a"],
    #                   "uv": ["UV254", "E250/E365", "E350/E400", "S275-295", "S350-400", "SR", "Chl-a"],
    #                   "uv_optics": ["UV254", "E250/E365", "E350/E400", "S275-295", "S350-400", "SR", "C1", "C2", "C3", "Chl-a"],
    #                   "fluorescence": ["FI450", "FI470", "BIX", "HIX", "Chl-a"],
    #                   "fluorescence_optics": ["FI450", "FI470", "BIX", "HIX", "C1", "C2", "C3", "Chl-a"],
    #                   "all": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "Chl-a", "UV254", "E250/E365", "E350/E400", "S275-295", "S350-400", "SR", "FI450", "FI470", "BIX", "HIX", "Chl-a"],
    #                   "all_optics": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "Chl-a", "UV254", "E250/E365", "E350/E400", "S275-295", "S350-400", "SR", "FI450", "FI470", "BIX", "HIX", "C1", "C2", "C3", "Chl-a"],
    #                   "optics": ["C1", "C2", "C3", "Chl-a"]
    #                  }
    list_features = {
        # "water": ["Water Temp.", "DOC", "BOD5", "CODMn", "SS", "Chl-a"],
        # "water_optics": ["Water Temp.", "DOC", "BOD5", "CODMn", "SS", "C2", "D2",  "Chl-a"],
        # "uv": ["UV254", "E250/E365", "E350/E400", "S275-295", "SR", "Chl-a"],
        # "uv_optics": ["UV254", "E250/E365", "E350/E400", "S275-295", "SR", "D1", "D2", "D3", "Chl-a"],
        # "fluorescence": ["FI450", "FI470", "BIX", "HIX", "Chl-a"],
        # "fluorescence_optics": ["FI450", "FI470", "BIX", "HIX", "C2", "D2", "Chl-a"],
        # "all": ["Water Temp.", "DOC", "BOD5", "CODMn", "SS", "UV254", "E250/E365", "E350/E400", "S275-295", "SR",  "FI450", "FI470", "BIX", "HIX", "Chl-a"],
        # "all_optics": ["Water Temp.", "DOC", "BOD5", "CODMn", "SS", "Chl-a", "UV254", "E250/E365", "E350/E400", "S275-295", "SR",  "FI450", "FI470", "BIX", "HIX", "C2", "D2",  "Chl-a"],
        # "optics": ["C1", "C2", "C3", "Chl-a"],
        # "fi_xgboost": ["SS", "CODMn", "BOD5", "DOC", "Chl-a"],
        # "fi_xgboost_optics": ["SS", "CODMn", "BOD5", "DOC", "C2", "D2", "Chl-a"],
        "fi_corr": ["Water Temp.", "pH", "DOC", "BOD5", "CODMn", "EC", "SS", "UV254", "Chl-a"],
        "fi_corr_optics": ["Water Temp.", "pH", "DOC", "BOD5", "CODMn", "EC", "SS", "UV254", "D2", "Chl-a"]
    }
    # models = ["linear_regression", "ridge_regression",
    #           "lasso_regression", "lasso_lars_regression", "bayesian_ridge_regression", "generalized_linear_regression", "kernel_ridge", "svm_regression", "KNeighbor_regression", "PLSRegression", "decision_tree_regression", "stacking"]
    models = ["linear_regression"]
    print(f'Total model using: {len(models)}')
    final_result = pd.DataFrame()
    final_result["model"] = models
    use_kfold = True
    for case, features in list_features.items():
        dataset = pd.read_csv('data/clo.csv', usecols=features)
        dataset = dataset.dropna()
        scaler = MinMaxScaler()
        scaler = scaler.fit(dataset)
        dataset = scaler.transform(dataset)
        train_per = 0.8
        size_split = int(len(dataset)*train_per)
        dataset_tna = dataset
        shuffle(dataset_tna)
        print(f'{size_split}')
        train_dataset = dataset_tna[:size_split]
        test_dataset = dataset_tna[size_split:]
        # train_dataset, test_dataset = train_test_split_custom(
        #     dataset, split=train_per)
        pprint(test_dataset)
        X_test = test_dataset[:, :-1]
        y_test = test_dataset[:, -1]
        results = {model: {"mae_train": [], "r2_train": [],
                           "mae_test": [], "r2_test": []} for model in models}
        if use_kfold:
            kfold = KFold(3, random_state=50, shuffle=True)
            x_trains = []
            y_trains = []
            for train_indexes, test_indexes in kfold.split(train_dataset):
                train_data = train_dataset[train_indexes]
                X_train = train_data[:, :-1]
                y_train = train_data[:, -1]
                x_trains.append(X_train)
                y_trains.append(y_train)
            for model in models:
                mae_train, r2_train, mae_test, r2_test = get_engine(
                    model)(x_trains, y_trains, X_test, y_test, scaler)
                append_result(model, mae_train, r2_train,
                              mae_test, r2_test)
        #!TODO làm sau
        # else:
        #     X_train, y_train, X_valid, y_valid, X_test, y_test, scaler = split_data(
        #         dataset)
        #     for model in models:
        #         mae_train, r2_train, mae_test, r2_test = get_engine(
        #             model)(X_train, y_train, X_test, y_test, scaler)
        #         append_result(model, mae_train, r2_train, mae_test, r2_test)
        r2_test = []
        for model in results:
            r2_test.append(np.mean(results[model]["r2_test"]))
        final_result[case] = r2_test
    final_result.to_csv("final_result.csv", index=False)
