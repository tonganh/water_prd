import pandas as pd
from common import seed_everything, run_xgboost, run_stacking, run_dl, run_stacking_v2

if __name__ == "__main__":
    seed_everything(42)
    dataset = pd.read_csv('data/clo.csv', usecols=["D2", "C2", "SS", "CODMn", "BOD5", "DOC", "Chl-a"])
    # dataset = pd.read_csv('data/clo.csv', usecols=["Water Temp.", "pH", "DOC", "BOD5", "CODMn", "EC", "SS", "UV254", "C1", "C2", "D1", "D2", "D3"])

    dataset = dataset.dropna()

    # run_xgboost(dataset)
    # run_dl(dataset)
    # run_stacking(dataset)
    run_stacking_v2(dataset)