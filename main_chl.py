import pandas as pd
from common import seed_everything, run_xgboost, run_stacking, run_dl, run_stacking_v2

if __name__ == "__main__":
    seed_everything(42)
    dataset = pd.read_csv('data/clo.csv', usecols=["D2", "C2", "SS", "CODMn", "BOD5", "DOC", "Chl-a"])
    dataset = dataset.dropna()
    # run_xgboost(dataset)
    # run_dl(dataset)
    # run_stacking(dataset)
    run_stacking_v2(dataset)