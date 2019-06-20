'''

Covers core aspects of Week 1 - Week 3 of Andrew Ng's Coursera Course
Topic: Linear Regression with Regularization
Data: https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data
Do change the PATH before running

'''
PATH = "/Users/pranjal/Google Drive/python_projects/projects/courses/andrew_ng/machine_learning/algorithms/linear_regression/superconductor.csv"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
np.random.seed(42)


def import_df(PATH):
    df = pd.read_csv(PATH)
    df = pd.concat([df.critical_temp, df.drop('critical_temp', axis=1)], axis=1)
    df = np.array(df, dtype='float')
    return df


def Xy(df):
    X = df[:, 1:]
    y = df[:, 0]
    y = y.reshape(y.shape[0], 1)
    return X, y


def featureScale(X):
    temp0 = X - X.mean(axis=0)
    temp1 = X.max(axis=0) - X.min(axis=0)
    return np.divide(temp0, temp1)


def randomTheta(X):
    m = X.shape[0]  # training examples
    n = X.shape[1] + 1  # parameters
    θ = np.random.normal(0, 1, (n, 1))
    return θ


def hypothesis(X, θ):
    m = X.shape[0]  # training examples
    ones = np.ones((m, 1))
    Xd = np.hstack((ones, X))
    h = np.dot(Xd, θ)
    return h, Xd


def squaredLoss(X, y, θ, λ):
    m = X.shape[0]  # training examples
    h, Xd = hypothesis(X, θ)
    e = h - y
    θ_temp = θ.copy()
    θ_temp[0] = 1
    J = sum(np.multiply(e, e)) + λ * sum(np.multiply(θ_temp, θ_temp))
    J = J[0] / (2 * m)
    return J


def gradient_descent(X, y, eval_set, θ, λ, iterations, α, mom):
    X_val, y_val = eval_set[0], eval_set[1]
    m = X.shape[0]  # training examples
    delta = 0
    cost_path = []
    eval_path = []
    for i in range(iterations):
        cost_path.append([squaredLoss(X, y, θ, λ), squaredLoss(X_val, y_val, θ, λ)])
        h, Xd = hypothesis(X, θ)
        h_val, Xd_val = hypothesis(X_val, θ)
        eval_path.append([r2_score(y, h), r2_score(y_val, h_val)])
        θ_gradient = np.dot(Xd.T, h - y) + λ * θ
        θ_gradient = θ_gradient / m
        delta = mom * delta + θ_gradient
        θ= θ - α * delta
    return θ, cost_path, eval_path


def train_test_split(X, y, size):
    ''' X_train, X_test, y_train, y_test '''
    df = np.hstack((y, X))
    np.random.shuffle(df)
    X, y = Xy(df)
    m = round(y.shape[0] * size)
    return X[m:, :], X[0:m, :], y[m:, :], y[0:m, :]


PATH = "/Users/pranjal/Google Drive/python_projects/projects/courses/andrew_ng/machine_learning/algorithms/linear_regression/superconductor.csv"
df = import_df(PATH)
X, y = Xy(df)
X = featureScale(X)

θ=randomTheta(X)
λ=np.random.normal(0, 1, 1)[0]
print(squaredLoss(X, y, θ, λ))

X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)
print(X_train.shape, X_test.shape)

θ, cost_path, eval_path = gradient_descent(X_train, y_train,
                                           [X_test, y_test], θ, 10, 1000, 0.1, 0.9)

plt.subplot(1, 2, 1)
cost_train = [i[0] for i in cost_path]
cost_val = [i[1] for i in cost_path]
plt.plot(cost_train, label='Cost Train')
plt.plot(cost_val, label='Cost Val')
plt.legend()
plt.title('Learning Curve - Cost')

plt.subplot(1, 2, 2)
R2_train = [i[0] for i in eval_path]
R2_val = [i[1] for i in eval_path]
plt.plot(R2_train, label='R2 Train')
plt.plot(R2_val, label='R2 Val')
plt.legend()
plt.title('Learning Curve - Eval Metric')
plt.show()
