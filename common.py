from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

def split_data(dataset, train_per=0.6, valid_per=0.0):
    # dataset = dataset.to_numpy()
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

    return X, Y, X_train, y_train, X_valid, y_valid, X_test, y_test

def evaluate(gt, pred, algo, phase_name):
    output_log = os.path.join("log", "visualization")
    if not os.path.exists(output_log):
        os.makedirs(output_log)
    mae = mean_absolute_error(gt, pred)
    r2 = r2_score(gt, pred)
    plt.figure(figsize=(16, 7))
    plt.plot(gt, label="gt")
    plt.plot(pred, label="pred")
    plt.legend()
    plt.savefig(os.path.join(output_log, "pred_{}_{}.png".format(algo, phase_name)))
    plt.close()
    print("MAE_{}: {:.3f}".format(phase_name, mae))
    print("R2_{}: {:.3f}".format(phase_name, r2))
    return mae, r2