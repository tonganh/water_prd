import pandas as pd
from common import seed_everything, run_xgboost, run_stacking, run_dl, run_stacking_v2

if __name__ == "__main__":
    seed_everything(42)
    dataset = pd.read_csv('data/hoat_chat.csv', usecols=["TOC", "UV254", "HS", "HAAFP", "THMFP"])
    target = "THMFP"
    target = "HAAFP"
    if target == "THMFP":
        dataset = dataset.drop(columns=["HAAFP"])
    else:
        dataset = dataset.drop(columns=["THMFP"])
    dataset = dataset.dropna()
    # run_xgboost(dataset)
    # run_dl(dataset)
    run_stacking(dataset)