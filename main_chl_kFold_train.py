from pprint import pprint
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from common import seed_everything, split_data
import warnings
import numpy as np
import importlib
from random import random, randrange, shuffle, sample
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
    # list_features = {
    #     # "water": ["Water Temp.", "DOC", "BOD5", "CODMn", "SS", "Chl-a"],
    #     # "water_optics": ["Water Temp.", "DOC", "BOD5", "CODMn", "SS", "C2", "D2",  "Chl-a"],
    #     # "uv": ["UV254", "E250/E365", "E350/E400", "S275-295", "SR", "Chl-a"],
    #     # "uv_optics": ["UV254", "E250/E365", "E350/E400", "S275-295", "SR", "D1", "D2", "D3", "Chl-a"],
    #     # "fluorescence": ["FI450", "FI470", "BIX", "HIX", "Chl-a"],
    #     # "fluorescence_optics": ["FI450", "FI470", "BIX", "HIX", "C2", "D2", "Chl-a"],
    #     # "all": ["Water Temp.", "DOC", "BOD5", "CODMn", "SS", "UV254", "E250/E365", "E350/E400", "S275-295", "SR",  "FI450", "FI470", "BIX", "HIX", "Chl-a"],
    #     # "all_optics": ["Water Temp.", "DOC", "BOD5", "CODMn", "SS", "Chl-a", "UV254", "E250/E365", "E350/E400", "S275-295", "SR",  "FI450", "FI470", "BIX", "HIX", "C2", "D2",  "Chl-a"],
    #     # "optics": ["C1", "C2", "C3", "Chl-a"],
    #     # "fi_xgboost": ["SS", "CODMn", "BOD5", "DOC", "Chl-a"],
    #     # "fi_xgboost_optics": ["SS", "CODMn", "BOD5", "DOC", "C2", "D2", "Chl-a"],
    #     "fi_corr": ["Water Temp.", "pH", "DOC", "BOD5", "CODMn", "EC", "SS", "Chl-a"],
    #     "fi_corr_optics": ["Water Temp.", "pH", "DOC", "BOD5", "CODMn", "EC", "SS", "UV254", "D2", "Chl-a"]
    # }
    list_features = {
        # "water": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "Chl-a"],
        # "water_1": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "Chl-a"],
        # "water_uv": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "UV254", "E250/E365", "E350/E400", "S275-295", "S350-400", "SR", "Chl-a"],
        # "water_uv_fluorence": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "UV254", "E250/E365", "E350/E400", "S275-295", "S350-400", "SR", "FI450", "FI470", "BIX", "HIX", "Chl-a"],
        # "some_water_uv": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "EC", "SS",  "UV254", "E250/E365", "E350/E400", "S275-295", "S350-400", "SR", "Chl-a"],
        # "water_get_some_uv": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "S275-295", "S350-400", "UV254", "Chl-a"],
        # "water_fluorence": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "FI450", "FI470", "BIX", "HIX", "Chl-a"],
        # "water_some_fluorence": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS",  "HIX", "Chl-a"],
        "water_some_fluorence_uv": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "S275-295", "S350-400", "UV254", "HIX", "Chl-a"],
        # "water_optics_c1_c2_c3": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "C1", "C2", "C3",  "Chl-a"],
        # "water_some_c1_c2_c3": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "C1", "C2",  "Chl-a"],
        # "some_water_c1_c2_c3": ["Water Temp.", "DO", "DOC", "BOD5", "CODMn",  "EC", "SS",  "C1", "C2", "C3", "Chl-a"],
        # "water_some_c1_c2_c3_fluorence": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "FI450", "FI470", "HIX", "C1", "C2", "C3", "Chl-a"],
        # "water_optics_d1_d2_d3_d4": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "D1", "D2", "D3", "D4", "Chl-a"],
        # "water_some_d1_d2_d3_d4_fluorence": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "FI450", "FI470", "BIX", "HIX", "D1", "D2", "D3", "D4", "Chl-a"],
        # "some_water_d1_d2_d3_d4": ["Water Temp.", "DO", "DOC",  "EC", "SS",  "D1", "D2", "D3",  "Chl-a"],
        # "uv": ["UV254", "E250/E365", "E350/E400", "S275-295", "SR", "Chl-a"],
        # "uv_optics": ["UV254", "E250/E365", "E350/E400", "S275-295", "SR", "D1", "D2", "D3", "Chl-a"],
        # "fluorescence": ["FI450", "FI470", "BIX", "HIX", "Chl-a"],
        # "fluorescence_optics": ["FI450", "FI470", "BIX", "HIX", "C2", "D2", "Chl-a"],
        # "all": ["Water Temp.", "DOC", "BOD5", "CODMn", "SS", "UV254", "E250/E365", "E350/E400", "S275-295", "SR",  "FI450", "FI470", "BIX", "HIX", "Chl-a"],

        # "all_optics": ["Water Temp.", "DOC", "BOD5", "CODMn", "SS", "Chl-a", "UV254", "E250/E365", "E350/E400", "S275-295", "SR",  "FI450", "FI470", "BIX", "HIX", "C2", "D2",  "Chl-a"],
        # "optics": ["C1", "C2", "C3", "Chl-a"],
        # "fi_xgboost": ["SS", "CODMn", "BOD5", "DOC", "Chl-a"],
        # "fi_xgboost_optics": ["SS", "CODMn", "BOD5", "DOC", "C2", "D2", "Chl-a"],
        # "fi_corr": ["Water Temp.", "pH", "DOC", "BOD5", "CODMn", "EC", "SS", "UV254", "Chl-a"],
        # "fi_corr_optics": ["Water Temp.", "pH", "DOC", "BOD5", "CODMn", "EC", "SS", "UV254", "D2", "Chl-a"]
    }
    # models = ["linear_regression", "ridge_regression",
    #           "lasso_regression", "lasso_lars_regression", "bayesian_ridge_regression", "generalized_linear_regression", "kernel_ridge", "svm_regression", "KNeighbor_regression", "PLSRegression", "decision_tree_regression", "stacking"]
    models = ["linear_regression", "xgboost"]
    print(f'Total model using: {len(models)}')
    use_kfold = True
    range_using = 20
    test_list = []
    for test_th in range(range_using):
        final_result = pd.DataFrame()
        final_result["model"] = models
        dataset = pd.read_csv('data/clo.csv')
        dataset = dataset.dropna()
        dataset = dataset.drop(
            columns=["Date", "No", "Sample No.", "Site", "Water Depth"])
        dataset = dataset.sample(frac=1)
        for case, features in list_features.items():
            case_dataset = dataset[features].copy()
            scaler = MinMaxScaler()
            scaler = scaler.fit(case_dataset)
            case_dataset = scaler.transform(case_dataset)
            train_per = 0.8
            size_split = int(len(case_dataset)*train_per)
            n_train = int(len(case_dataset)*0.8)
            train_dataset = case_dataset[:size_split]
            test_dataset = case_dataset[size_split:]

            # train_dataset, test_dataset = train_test_split_custom(
            #     dataset, split=train_per)
            X_test = test_dataset[:, :-1]
            y_test = test_dataset[:, -1]
            results = {model: {"mae_train": [], "r2_train": [],
                               "mae_test": [], "r2_test": []} for model in models}
            if use_kfold:
                kfold = KFold(5, random_state=50, shuffle=True)
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
            test_list.append(r2_test)
        final_result.to_csv("final_result_{}.csv".format(test_th), index=False)
        print(test_th)
    linear_count = 0
    xgboost = 0
    max_linear = -1000
    min_linear = 1000
    max_xgboost = -1000
    min_xgboost = 1000
    for each_couple_value in test_list:
        [linear_value, xgboost_value] = each_couple_value
        if linear_value > max_linear:
            max_linear = linear_value
        if min_linear > linear_value:
            min_linear = linear_value

        if xgboost_value > max_xgboost:
            max_xgboost = xgboost_value
        if min_xgboost > xgboost_value:
            min_xgboost = xgboost_value

        linear_count = linear_count + linear_value
        xgboost = xgboost + xgboost_value
    avg_linear = linear_count/range_using
    avg_xgboost = xgboost/range_using
    print(f'linear: min: {min_linear} - avg: {avg_linear} ma: {max_linear}')
    print(
        f'xgboost: min: {min_xgboost} - avg: {avg_xgboost} ma: {max_xgboost}')
