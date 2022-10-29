from pprint import pprint
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from common import seed_everything, split_data
import warnings
import numpy as np
import importlib
warnings.filterwarnings(
    action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')


def append_result(model, mae_train, r2_train, mae_test, r2_test):
    # results[model]["mae_train"].append(mae_train)
    # results[model]["r2_train"].append(r2_train)
    results[model]["mae_test"].append(mae_test)
    results[model]["r2_test"].append(r2_test)


def get_engine(model):
    package = importlib.import_module('common')
    model_engine = getattr(package, "run_{}".format(model))
    return model_engine


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
        "water": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "Chl-a"],
        "water_uv": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "UV254", "E250/E365", "E350/E400", "S275-295", "S350-400", "SR", "Chl-a"],
        "water_fluorence": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "FI450", "FI470", "BIX", "HIX", "Chl-a"],
        "water_optics_c1_c2_c3": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "C1", "C2", "C3",  "Chl-a"],
        "water_optics_d1_d2_d3_d4": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "D1", "D2", "D3", "D4", "Chl-a"],
        "water_get_some_uv": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "S275-295", "S350-400", "UV254", "Chl-a"],
        "water_some_fluorence": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS",  "HIX", "Chl-a"],
        "water_some_c1_c2_c3": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "C1", "C2",  "Chl-a"],
        "water_some_c1_c2_c3_fluorence": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "FI450", "FI470", "HIX", "C1", "C2", "C3", "Chl-a"],
        "water_some_d1_d2_d3_d4_fluorence": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "DTP", "EC", "SS", "FI450", "FI470", "BIX", "HIX", "D1", "D2", "D3", "D4", "Chl-a"],
        "some_water_uv": ["Water Temp.", "pH", "DO", "DOC", "BOD5", "CODMn", "DTN", "EC", "SS",  "UV254", "E250/E365", "E350/E400", "S275-295", "S350-400", "SR", "Chl-a"],
        "some_water_c1_c2_c3": ["Water Temp.", "DO", "DOC", "BOD5", "CODMn",  "EC", "SS",  "C1", "C2", "C3", "Chl-a"],
        "some_water_d1_d2_d3_d4": ["Water Temp.", "DO", "DOC", "BOD5", "CODMn",  "EC", "SS",  "D1", "D2", "D3",  "Chl-a"],
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
    # models = ["linear_regression", "ridge_regression", "bayesian_ridge_regression", "generalized_linear_regression",
    #           "kernel_ridge",  "KNeighbor_regression", "PLSRegression",  "stacking"]
    # models = ["linear_regression",  "bayesian_ridge_regression",
    #           "stacking"]
    models = ["linear_regression"]
    print(f'Total model using: {len(models)}')
    final_result = pd.DataFrame()
    final_result["model"] = models
    use_kfold = True
    for case, features in list_features.items():
        import pdb
        dataset = pd.read_csv('data/clo.csv', usecols=features)
        dataset = dataset.dropna()
        scaler = MinMaxScaler()
        scaler = scaler.fit(dataset)
        dataset = scaler.transform(dataset)

        results = {model: {"mae_train": [], "r2_train": [],
                           "mae_test": [], "r2_test": []} for model in models}
        if use_kfold:
            kfold = KFold(5, random_state=50, shuffle=True)
            for train_indexes, test_indexes in kfold.split(dataset):
                train_data = dataset[train_indexes]
                test_data = dataset[test_indexes]
                X_train = train_data[:, :-1]
                y_train = train_data[:, -1]
                X_test = test_data[:, :-1]
                y_test = test_data[:, -1]
                for model in models:
                    mae_train, r2_train, mae_test, r2_test = get_engine(
                        model)(X_train, y_train, X_test, y_test, scaler)
                    append_result(model, mae_train, r2_train,
                                  mae_test, r2_test)
        else:
            X_train, y_train, X_valid, y_valid, X_test, y_test, scaler = split_data(
                dataset)
            for model in models:
                mae_train, r2_train, mae_test, r2_test = get_engine(
                    model)(X_train, y_train, X_test, y_test, scaler)
                append_result(model, mae_train, r2_train, mae_test, r2_test)
        r2_test = []
        for model in results:
            r2_test.append(np.mean(results[model]["r2_test"]))
        final_result[case] = r2_test
        if case != 'water':
            test = (r2_test / final_result['water'])*100
            print(f'{test}')
            case_add_percentage = case + '_up_than_water'
            final_result[case_add_percentage] = f'{round(float(test),2)}%'

    final_result.to_csv("final_result.csv", index=False)
