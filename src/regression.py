import numpy as np


def hypothesis(X, theta):
    return X @ theta


def cost(X, y, theta):
    error = hypothesis(X, theta) - y
    return 0.5 * np.sum(error ** 2)


def fit_normal(X, y):
    return np.linalg.inv(X.T @ X) @ (X.T @ y)


def fit_batch_gd(X, y, alpha, n_iters):
    n_features = X.shape[1]
    theta = np.zeros(n_features)
    history = np.zeros(n_iters)
    for t in range(n_iters):
        error = y - hypothesis(X, theta)
        theta = theta + alpha * (X.T @ error)
        history[t] = cost(X, y, theta)
    return theta, history


def fit_sgd(X, y, alpha, n_epochs):
    n_rows, n_features = X.shape
    theta = np.zeros(n_features)
    history = np.zeros(n_epochs)
    for epoch in range(n_epochs):
        for i in range(n_rows):
            xi = X[i]
            yi = y[i]
            hi = xi @ theta
            theta = theta + alpha * (yi - hi) * xi
        history[epoch] = cost(X, y, theta)
    return theta, history


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def clip_nonnegative(predictions):
    return np.maximum(predictions, 0.0)
