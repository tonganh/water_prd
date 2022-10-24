import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from common import seed_everything, run_xgboost, run_stacking, run_dl, run_stacking_v2, split_data
import warnings
import numpy as np
warnings.filterwarnings(action="ignore", message=r'.*Use subset.*of np.ndarray is not recommended')

def append_result(model, mae_train, r2_train, mae_test, r2_test):
    results[model]["mae_train"].append(mae_train)
    results[model]["r2_train"].append(r2_train)
    results[model]["mae_test"].append(mae_test)
    results[model]["r2_test"].append(r2_test)

if __name__ == "__main__":
    seed_everything(42)
    dataset = pd.read_csv('data/clo.csv', usecols=["Water Temp.","pH","DO","DOC","BOD5","CODMn","DTN","DTP","EC","SS", "C1", "C2", "C3", "Chl-a"])
    dataset = dataset.dropna()
    scaler = MinMaxScaler()
    scaler = scaler.fit(dataset)
    dataset = scaler.transform(dataset)

    use_kfold = True
    if use_kfold:
        kfold = KFold(3, random_state=42, shuffle=True)
        results = {"xgboost": {"mae_train": [], "r2_train": [], "mae_test": [], "r2_test": []}, 
        "deeplearning": {"mae_train": [], "r2_train": [], "mae_test": [], "r2_test": []}, 
        "stacking": {"mae_train": [], "r2_train": [], "mae_test": [], "r2_test": []}}
        for train_indexes, test_indexes in kfold.split(dataset):
            train_data = dataset[train_indexes]
            test_data = dataset[test_indexes]
            X_train = train_data[:, :-1]
            y_train = train_data[:, -1]
            X_test = test_data[:, :-1]
            y_test = test_data[:, -1]
            # mae_train_xgb, r2_train_xgb, mae_test_xgb, r2_test_xgb = run_xgboost(X_train, y_train, X_test, y_test, scaler)
            # append_result("xgboost", mae_train_xgb, r2_train_xgb, mae_test_xgb, r2_test_xgb)
            # mae_train_dl, r2_train_dl, mae_test_dl, r2_test_dl = run_dl(X_train, y_train, X_test, y_test, scaler)
            # append_result("deeplearning", mae_train_dl, r2_train_dl, mae_test_dl, r2_test_dl)
            mae_train_stacking, r2_train_stacking, mae_test_stacking, r2_test_stacking = run_stacking_v2(X_train, y_train, X_test, y_test, scaler)
            append_result("stacking", mae_train_stacking, r2_train_stacking, mae_test_stacking, r2_test_stacking )
    else:
        X_train, y_train, X_valid, y_valid, X_test, y_test, scaler = split_data(dataset)
        mae_train_xgb, r2_train_xgb, mae_test_xgb, r2_test_xgb = run_xgboost(X_train, y_train, X_test, y_test, scaler)
        append_result("xgboost", mae_train_xgb, r2_train_xgb, mae_test_xgb, r2_test_xgb)
        mae_train_dl, r2_train_dl, mae_test_dl, r2_test_dl = run_dl(X_train, y_train, X_test, y_test, scaler)
        append_result("deeplearning", mae_train_dl, r2_train_dl, mae_test_dl, r2_test_dl)
        mae_train_stacking, r2_train_stacking, mae_test_stacking, r2_test_stacking = run_stacking_v2(X_train, y_train, X_test, y_test, scaler)
        append_result("stacking", mae_train_stacking, r2_train_stacking, mae_test_stacking, r2_test_stacking )
    for model in results:
        print("Model: ", model)
        print("mae_train: {}".format(np.mean(results[model]["mae_train"])))
        print("mae_test: {}".format(np.mean(results[model]["mae_test"])))
        print("r2_train: {}".format(np.mean(results[model]["r2_train"])))
        print("r2_test: {}".format(np.mean(results[model]["r2_test"])))
        print()