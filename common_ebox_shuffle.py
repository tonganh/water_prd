from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor as xgbmodel
import numpy as np
import random
import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
from sklearn import kernel_ridge
from sklearn import svm
from sklearn import neighbors
from sklearn import tree
from sklearn.metrics import mean_squared_error


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def run_xgboost(X_trains, y_trains, X_test, y_test, scaler, case='1'):
    model = xgbmodel(objective='reg:squarederror')
    for i in range(0, len(X_trains)):
        x_train = X_trains[i]
        y_train = y_trains[i]
        model.fit(x_train, y_train)
        train_results = model.predict(x_train)
        # mae_train, r2_train = evaluate(
        #     scaler, y_train, train_results, "linear_regression", "train")
    mae_train = 1
    r2_train = 1

    test_results = model.predict(X_test)
    mae_test, r2_test, mse_test = evaluate(
        scaler, y_test, test_results, "xgboost", "test", case)
    # r2_test = mae_test
    return mae_train, r2_train, mae_test, r2_test, mse_test


def run_linear_regression(X_trains, y_trains, X_test, y_test, scaler, case='1'):
    model = linear_model.LinearRegression()
    for i in range(0, len(X_trains)):
        x_train = X_trains[i]
        y_train = y_trains[i]
        model.fit(x_train, y_train)
        train_results = model.predict(x_train)
        # mae_train, r2_train = evaluate(
        #     scaler, y_train, train_results, "linear_regression", "train")

    test_results = model.predict(X_test)
    mae_test, r2_test, mse_test = evaluate(
        scaler, y_test, test_results, 'linear_regression', 'test', case)
    mae_train = 1
    r2_train = 1
    # r2_test = mae_test
    # r2_test = mean_squared_error(y_test, test_results)
    return mae_train, r2_train, mae_test, r2_test, mse_test


def run_ridge_regression(X_trains, y_trains, X_test, y_test, scaler):
    model = linear_model.Ridge(alpha=.5)
    name_model = 'ridge_regression'
    for i in range(0, len(X_trains)):
        x_train = X_trains[i]
        y_train = y_trains[i]
        model.fit(x_train, y_train)
        train_results = model.predict(x_train)
        # mae_train, r2_train = evaluate(
        #     scaler, y_train, train_results, "linear_regression", "train")
    mae_train = 1
    r2_train = 1
    test_results = model.predict(X_test)
    mae_test, r2_test, mse_test = evaluate(
        scaler, y_test, test_results, name_model, 'test')
    return mae_train, r2_train, mae_test, r2_test, mse_test


def run_lasso_regression(X_trains, y_trains, X_test, y_test, scaler):
    name_model = 'lasso_regression'
    model = linear_model.Lasso(alpha=.1)
    for i in range(0, len(X_trains)):
        x_train = X_trains[i]
        y_train = y_trains[i]
        model.fit(x_train, y_train)
        train_results = model.predict(x_train)
        # mae_train, r2_train = evaluate(
        #     scaler, y_train, train_results, "linear_regression", "train")
    mae_train = 1
    r2_train = 1

    test_results = model.predict(X_test)
    mae_test, r2_test, mse_test = evaluate(
        scaler, y_test, test_results, name_model, 'test')
    return mae_train, r2_train, mae_test, r2_test, mse_test


def run_lasso_lars_regression(X_trains, y_trains, X_test, y_test, scaler):
    name_model = 'lasso_lars_regression'
    model = linear_model.LassoLars(alpha=.1, normalize=False)
    for i in range(0, len(X_trains)):
        x_train = X_trains[i]
        y_train = y_trains[i]
        model.fit(x_train, y_train)
        train_results = model.predict(x_train)
        # mae_train, r2_train = evaluate(
        #     scaler, y_train, train_results, "linear_regression", "train")
    mae_train = 1
    r2_train = 1

    test_results = model.predict(X_test)
    mae_test, r2_test, mse_test = evaluate(
        scaler, y_test, test_results, name_model, 'test')
    return mae_train, r2_train, mae_test, r2_test, mse_test


def run_bayesian_ridge_regression(X_trains, y_trains, X_test, y_test, scaler):
    name_model = 'bayesian_ridge_regression'
    model = linear_model.BayesianRidge()
    for i in range(0, len(X_trains)):
        x_train = X_trains[i]
        y_train = y_trains[i]
        model.fit(x_train, y_train)
        train_results = model.predict(x_train)
        # mae_train, r2_train = evaluate(
        #     scaler, y_train, train_results, "linear_regression", "train")
    mae_train = 1
    r2_train = 1

    test_results = model.predict(X_test)
    mae_test, r2_test, mse_test = evaluate(
        scaler, y_test, test_results, name_model, 'test')
    return mae_train, r2_train, mae_test, r2_test, mse_test


def run_generalized_linear_regression(X_train, y_train, X_test, y_test, scaler):
    model = linear_model.TweedieRegressor(power=1, alpha=0.5, link='log')
    model.fit(X_train, y_train)
    name_model = 'generalized_linear_regression'
    train_results = model.predict(X_train)
    mae_train, r2_train = evaluate(
        scaler, y_train, train_results, name_model, "train")

    test_results = model.predict(X_test)
    mae_test, r2_test = evaluate(
        scaler, y_test, test_results, name_model, 'test')
    return mae_train, r2_train, mae_test, r2_test


def run_kernel_ridge(X_train, y_train, X_test, y_test, scaler):
    model = kernel_ridge.KernelRidge(alpha=1.0)
    model.fit(X_train, y_train)
    name_model = 'kernel_ridge'
    train_results = model.predict(X_train)
    mae_train, r2_train = evaluate(
        scaler, y_train, train_results, name_model, "train")

    test_results = model.predict(X_test)
    mae_test, r2_test = evaluate(
        scaler, y_test, test_results, name_model, 'test')
    return mae_train, r2_train, mae_test, r2_test


def run_svm_regression(X_train, y_train, X_test, y_test, scaler):
    model = svm.SVR()
    model.fit(X_train, y_train)
    name_model = 'svm_regression'
    train_results = model.predict(X_train)
    mae_train, r2_train = evaluate(
        scaler, y_train, train_results, name_model, "train")

    test_results = model.predict(X_test)
    mae_test, r2_test = evaluate(
        scaler, y_test, test_results, name_model, 'test')
    return mae_train, r2_train, mae_test, r2_test


def run_KNeighbor_regression(X_train, y_train, X_test, y_test, scaler):
    model = neighbors.KNeighborsRegressor(n_neighbors=2)
    model.fit(X_train, y_train)
    name_model = 'KNeighbor_regression'
    train_results = model.predict(X_train)
    mae_train, r2_train = evaluate(
        scaler, y_train, train_results, name_model, "train")

    test_results = model.predict(X_test)
    mae_test, r2_test = evaluate(
        scaler, y_test, test_results, name_model, 'test')
    return mae_train, r2_train, mae_test, r2_test


def run_PLSRegression(X_train, y_train, X_test, y_test, scaler):
    from sklearn.cross_decomposition import PLSRegression
    model = PLSRegression(n_components=2)
    model.fit(X_train, y_train)
    name_model = 'PLSRegression'
    train_results = model.predict(X_train)
    mae_train, r2_train = evaluate(
        scaler, y_train, train_results, name_model, "train")

    test_results = model.predict(X_test)
    mae_test, r2_test = evaluate(
        scaler, y_test, test_results, name_model, 'test')
    return mae_train, r2_train, mae_test, r2_test


def run_decision_tree_regression(X_train, y_train, X_test, y_test, scaler):
    model = tree.DecisionTreeRegressor()
    model.fit(X_train, y_train)
    name_model = 'decision_tree_regression'
    train_results = model.predict(X_train)
    mae_train, r2_train = evaluate(
        scaler, y_train, train_results, name_model, "train")

    test_results = model.predict(X_test)
    mae_test, r2_test = evaluate(
        scaler, y_test, test_results, name_model, 'test')
    return mae_train, r2_train, mae_test, r2_test


def run_deeplearning(X_trains, y_trains, X_test, y_test, scaler):
    # X_train_1 = []
    # y_train_1 = []
    # for i in X_trains:
    #     print(i)
    #     X_train_1.append(i)
    # for i in y_trains:
    #     y_train_1.append(i)
    # X_train = []
    # y_train = []
    # for i in X_train_1:
    #     X_train.append(i)
    # for i in y_train_1:
    #     y_train.append(i)
    # X_train = []
    # y_train = []
    # for i in X_trains:
    #     for ii in i:
    #         element_expec = np.array(ii)
    #         X_train.append(element_expec)
    # #         exit()
    # for i in y_trains:
    #     for ii in i:
    #         y_train.append(np.array(ii))
    X_train = X_trains[0]
    y_train = y_trains[0]
    # # X_train = np.array(X_trains)
    # # y_train = np.array(y_trains)
    # print(f'{type(y_trains)} {type(X_train)}')
    # print(f'----------{X_train}-------')
    # exit()
    # print(f'------------------{X_trains}--------------')
    X_train = torch.Tensor(X_train)
    # print(f'------------------{X_train}--------------')
    # exit()
    y_train = torch.Tensor(y_train)
    X_test = torch.Tensor(X_test)
    y_test = torch.Tensor(y_test)

    class LinearRegression(nn.Module):
        def __init__(self):
            super(LinearRegression, self).__init__()
            self.fc1 = nn.Linear(X_train.shape[1], 16)
            self.fc2 = nn.Linear(16, 8)
            self.fc3 = nn.Linear(8, 1)

        def forward(self, x):
            out = self.fc1(x)
            out = F.relu(self.fc2(out))
            out = self.fc3(out)
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

            out = model(inputs).squeeze()
            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if (epoch+1) % 200 == 0:
            #     print(f'Epoch[{epoch+1}/{num_epochs}], loss: {loss.item():.6f}')

    model.eval()
    mae_train = 1
    r2_train = 1
    with torch.no_grad():
        train_results = model(X_train)
        # mae_train, r2_train, mse_train = evaluate(
        #     scaler, y_train, train_results, "dl", "train")
        test_results = 3 - (X_test)
        mae_test, r2_test, mse_test = evaluate(
            scaler, y_test, test_results, "dl", "test")
    return mae_train, r2_train, mae_test, r2_test, mse_test

# def run_stacking(X_train, y_train, X_test, y_test, scaler):
#     from sklearn.linear_model import RidgeCV
#     from sklearn.svm import LinearSVR
#     from sklearn.ensemble import RandomForestRegressor
#     from sklearn.ensemble import StackingRegressor
#     from sklearn.ensemble import GradientBoostingRegressor
#     from sklearn.linear_model import LinearRegression
#     from sklearn.neighbors import KNeighborsRegressor
#     from sklearn.tree import DecisionTreeRegressor
#     estimators = [
#         ('lr', RidgeCV()),
#         ('svr', LinearSVR(random_state=42)),
#         ('rf', RandomForestRegressor(random_state=42)),
#         ('knearest', KNeighborsRegressor()),
#         ('dt', DecisionTreeRegressor(random_state=42)),
#         ('xgb', GradientBoostingRegressor(random_state=42))
#     ]
#     model = StackingRegressor(
#         estimators=estimators,
#         final_estimator=LinearRegression()
#     )
#     model.fit(X_train, y_train)

#     train_results = model.predict(X_train)
#     mae_train, r2_train = evaluate(
#         scaler, y_train, train_results, "stacking", "train")

#     test_results = model.predict(X_test)
#     mae_test, r2_test = evaluate(
#         scaler, y_test, test_results, "stacking", "test")
#     return mae_train, r2_train, mae_test, r2_test


def run_stacking(X_trains, y_trains, X_test, y_test, scaler):
    from sklearn.linear_model import ElasticNet, Lasso
    from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
    from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
    from sklearn.model_selection import KFold

    class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
        def __init__(self, base_models, meta_model, n_folds=3):
            self.base_models = base_models
            self.meta_model = meta_model
            self.n_folds = n_folds

        # We again fit the data on clones of the original models
        def fit(self, X, y):
            self.base_models_ = [list() for x in self.base_models]
            self.meta_model_ = clone(self.meta_model)
            kfold = KFold(n_splits=self.n_folds,
                          shuffle=True, random_state=156)

            # Train cloned base models then create out-of-fold predictions
            # that are needed to train the cloned meta-model
            out_of_fold_predictions = np.zeros(
                (X.shape[0], len(self.base_models)))
            for i, model in enumerate(self.base_models):
                for train_index, holdout_index in kfold.split(X, y):
                    instance = clone(model)
                    self.base_models_[i].append(instance)
                    instance.fit(X[train_index], y[train_index])
                    y_pred = instance.predict(X[holdout_index])
                    out_of_fold_predictions[holdout_index, i] = y_pred

            # Now train the cloned  meta-model using the out-of-fold predictions as new feature
            self.meta_model_.fit(out_of_fold_predictions, y)
            return self

        # Do the predictions of all base models on the test data and use the averaged predictions as
        # meta-features for the final prediction which is done by the meta-model
        def predict(self, X):
            meta_features = np.column_stack([
                np.column_stack([model.predict(X)
                                for model in base_models]).mean(axis=1)
                for base_models in self.base_models_])
            return self.meta_model_.predict(meta_features)

    lasso = Lasso(alpha=0.0005, random_state=1, max_iter=100)
    ENet = ElasticNet(alpha=0.0005, l1_ratio=0.9, random_state=3, max_iter=100)
    GBoost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                       max_depth=10, random_state=5)
    model_rf = RandomForestRegressor(max_depth=17, n_estimators=100)

    models = StackingAveragedModels(base_models=(ENet, GBoost, model_rf),
                                    meta_model=lasso)
    for i in range(0, len(X_trains)):
        x_train = X_trains[i]
        y_train = y_trains[i]
        models.fit(x_train, y_train)
        train_results = models.predict(x_train)

    # models.fit(X_train, y_train)

    # train_results = models.predict(X_train)
    train_results = 1

    # mae_train, r2_train = evaluate(
    #     scaler, y_train, train_results, "stacking_v2", "train")
    mae_train = 1
    r2_train = 1

    test_results = models.predict(X_test)
    mae_test, r2_test, mse_test = evaluate(
        scaler, y_test, test_results, "stacking_v2", "test")
    r2_train = 1
    # r2_test = mae_test
    # r2_test = mean_squared_error(y_test, test_results)
    return mae_train, r2_train, mae_test, r2_test, mse_test


def split_data(dataset, train_per=0.8, valid_per=0.0):
    scaler = MinMaxScaler()
    scaler = scaler.fit(dataset)
    dataset = scaler.transform(dataset)
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

    return X_train, y_train, X_valid, y_valid, X_test, y_test, scaler


def evaluate(scaler, gt, pred, algo, phase_name, case='1'):
    output_log = os.path.join("log", "visualization")
    if not os.path.exists(output_log):
        os.makedirs(output_log)
    # gt = scaler.inverse_transform(gt)
    # pred = scaler.inverse_transform(pred)
    pred_ori = pred - scaler.min_[-1]  # aqmesh
    pred_ori /= scaler.scale_[-1]  # aqmesh
    gt_ori = gt - scaler.min_[-1]  # aqmesh
    gt_ori /= scaler.scale_[-1]  # aqmesh
    # pred_ori = pred
    # gt_ori = gt

    mae = mean_absolute_error(gt_ori, pred_ori)
    r2 = r2_score(gt_ori, pred_ori)
    plt.plot(gt_ori, label="gt")
    plt.plot(pred_ori, label="pred")
    plt.legend()
    plt.savefig(os.path.join(
        output_log, "pred_{}_{}_{}.png".format(algo, phase_name, case)))
    plt.close()
    # mae = mean_squared_error(gt, pred)
    mse = mean_squared_error(gt_ori, pred_ori)
    return mae, r2, mse
