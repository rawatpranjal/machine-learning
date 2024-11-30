'''
Covers core aspects of Andrew Ng's Coursera Course
Topic: Week 1 - Week 2: Linear Regression
Data: CSV from https://archive.ics.uci.edu/ml/machine-learning-databases/00464/
Problem: Regression
Features = 81 Physical Attributes (All Numerical)
Observations = 21k+
'''

PATH = "/Users/pranjal/Google Drive/python_projects/projects/courses/andrew_ng/machine_learning/algorithms/linear_regression/superconductor.csv"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
np.random.seed(42)

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")


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


def analyticalSolution(X, y):
    '''Unregularized Analytical Solution'''
    m = X.shape[0]  # training examples
    ones = np.ones((m, 1))
    Xd = np.hstack((ones, X))
    P = np.dot(Xd.T, Xd)
    P = np.linalg.inv(P)
    θ = np.dot(np.dot(P, Xd.T), y)
    return θ


# Load the data
print('\n\n')
print('----LINEAR REGRESSION MODEL----')

print('\n\n')
print('----LOAD DATA----')
df = import_df(PATH)
X, y = Xy(df)
print('X shape: ', X.shape)
print('y shape: ', y.shape)

print('\n\n')
print('----PREPROCESSING----')
X = featureScale(X)
print('Feature Scaling Completed!')

print('\n\n')
print('----INITIALISE----')
θ=randomTheta(X)
λ = 0
print('L2 Regularization: ', λ)
print('Random θ Vector (top 5 rows): \n', θ[0:5])
print(f'Cost at Initial θ: {squaredLoss(X, y, θ, λ):.2f}')

print('\n\n')
print('----TRAIN-TEST SPLIT----')
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.2)
print('X_train shape: ', X_train.shape)
print('X_test shape: ', X_test.shape)

# Gradient Descent
# Takes some time to run to get similar results at Sklearn/Analytical Soln
θ, cost_path, eval_path = gradient_descent(X_train, y_train, [X_test, y_test], θ, λ, 5000, 0.5, 0.9)

# Analytical Solution
θ_analy = analyticalSolution(X_train, y_train)
h_analy, Xd = hypothesis(X_test, θ_analy)

# Sklearn
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Comparision
print('\n\n')
print('----RESULTS COMPARISION----')
print(f'Test R2 from Sklearn: {r2_score(y_test, model.predict(X_test)):.4f}')
print(f'Test R2 from Gradient Descent: {eval_path[-1][1]: .4f}')
print(f'Test R2 from Analytical Soln: {r2_score(y_test, h_analy):.4f}')

# Plot Learning Curves for Gradient Descent Soln
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
