# from sklearn.metrics import recall_score
from xgboost import XGBRegressor as xgbmodel
import pandas as pd
import numpy as np
import random, os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from common import split_data, evaluate
from sklearn.preprocessing import MinMaxScaler
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
        
def run_xgboost(dataset):
    X, Y, X_train, y_train, _, _, X_test, y_test = split_data(dataset)
    # model = xgbmodel()
    model = xgbmodel(objective='reg:squarederror')
    # model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)])
    # model.fit(X_train, y_train, eval_metric="mae", eval_set=[(X_valid, y_valid)], verbose=False)
    model.fit(X_train, y_train, eval_metric="mae", verbose=False)

    all_results = model.predict(X)
    evaluate(Y, all_results, "xgboost", "all")

    train_results = model.predict(X_train)
    evaluate(y_train, train_results, "xgboost", "train")

    test_results = model.predict(X_test)
    evaluate(y_test, test_results, "xgboost", "test")


def run_dl(dataset):
    X, Y, X_train, y_train, _, _, X_test, y_test = split_data(dataset)
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    X_train = torch.Tensor(X_train)
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)
    class LinearRegression(nn.Module):
        def __init__(self):
            super(LinearRegression, self).__init__()
            self.fc1 = nn.Linear(3, 16)
            self.fc2 = nn.Linear(16, 8)
            self.fc3 = nn.Linear(8, 1)

        def forward(self, x):
            out = self.fc1(x)
            out = F.relu(self.fc2(out))
            out =self.fc3(out)
            return out

    model = LinearRegression()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    bs = 32
    num_epochs = 3000
    for epoch in range(num_epochs):
        for i in range(0, len(X_train), bs):
            inputs = X_train[i:i+bs]
            target = y_train[i:i+bs]

            out = model(inputs)
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (epoch+1) % 200 == 0:
            #     print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')

    model.eval()
    with torch.no_grad():
        all_results = model(X)
        evaluate(Y, all_results, "dl", "all")

        train_results = model(X_train)
        evaluate(y_train, train_results, "dl", "train")

        test_results = model(X_test)
        evaluate(y_test, test_results, "dl", "test")

def run_stacking(dataset):
    from sklearn.datasets import load_diabetes
    from sklearn.linear_model import RidgeCV
    from sklearn.svm import LinearSVR
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.ensemble import StackingRegressor
    from sklearn.ensemble import GradientBoostingRegressor
    # X, y = load_diabetes(return_X_y=True)
    X, Y, X_train, y_train, _, _, X_test, y_test = split_data(dataset)
    estimators = [
        ('lr', RidgeCV()),
        ('svr', LinearSVR(random_state=42)),
        ('xgb', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0))
    ]
    model = StackingRegressor(
        estimators=estimators,
        final_estimator=RandomForestRegressor(n_estimators=10,
                                            random_state=42)
    )
    # from sklearn.model_selection import train_test_split
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, random_state=42
    # )
    model.fit(X_train, y_train).score(X_test, y_test)
    all_results = model.predict(X)
    evaluate(Y, all_results, "stacking", "all")

    train_results = model.predict(X_train)
    evaluate(y_train, train_results, "stacking", "train")

    test_results = model.predict(X_test)
    evaluate(y_test, test_results, "stacking", "test")

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
    scaler = MinMaxScaler()
    scaler = scaler.fit(dataset)
    dataset = scaler.transform(dataset)
    # run_xgboost(dataset)
    # run_dl(dataset)
    run_stacking(dataset)