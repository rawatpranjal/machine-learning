'''
This version of the classic Neural Net develops the classic Neural Network. The inspiration
has been Andrew Ng's ML and DL course on Coursera.

Multiple Loss Functions:
->Binary Classification (Binary Logloss with Sigmoid), 
->Regression (RMSE Loss), 
->Multi-Classification (Cross-Entropy Loss with Softmax)

Improved Training:
->Mini-Batches,
->Careful Weight Initialization (He, Xavier),
->Adam Optimizer (Momentum + RMSProp)

'''


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def featureScaling(X, scaleType='standardize', axis=1):
    ε = 0.0000001
    X_max, X_min = np.max(X, axis=axis, keepdims=True), np.min(X, axis=axis, keepdims=True)
    X_mean, X_std = np.mean(X, axis=axis, keepdims=True), np.std(X, axis=axis, keepdims=True)
    scaler = {'min_max': (X - X_min) / (X_max - X_min + ε),
              'ng_scaler': (X - X_mean) / (X_max - X_min +ε),
              'standardize': (X - X_mean) / (X_std +ε)}
    return scaler[scaleType]


def miniBatchGenerator(m, size):
    number = [i for i in range(round(m / size))]
    cuts = []
    for i in number:
        cuts.append([i * size, i * size + size])
    cuts[-1][1] = m + 1
    return cuts


def dataSplit(X, y, split=[60, 20, 20], shuffle=True):
    a, b, c = split[0] / sum(split), split[1] / sum(split), split[2] / sum(split)
    if shuffle == True:
        df = np.r_[y, X]
        df = df.T
        np.random.shuffle(df)
        df = df.T
    y, X = df[0:y.shape[0], :].reshape(y.shape[0], X.shape[1]), df[y.shape[0]:, :]
    cut1, cut2 = round(X.shape[1] * a), round(X.shape[1] * b)
    return [X[:, 0:cut1], y[:, 0:cut1]], [X[:, cut1:cut1 + cut2], y[:, cut1:cut1 + cut2]], [X[:, cut1 + cut2:], y[:, cut1 + cut2:]]


def initialization(cache):
    default = {'cost_function': 'logloss', 'lrate': 0.1, 'iterations': 100, 'random_seed': 42,
               'hidden_layers': [10], 'activations': ['relu', 'sigmoid'], 'init': 'He', 'minibatchsize': 32,
               'X_train': x, 'y_train': y, 'metrics': ['accuracy'], 'momentum': 0.9, 'rmsprop': 0.999, 'epsilon': 0.00000001,
               'X_val': np.array([[]]), 'y_val': np.array([[]]), 'X_test': np.array([[]]), 'y_test': np.array([[]])}
    default.update(cache)
    cache = default.copy()

    cache['layers'] = [cache['X_train'].shape[0]] + cache['hidden_layers'] + [cache['y_train'].shape[0]]
    cache['metrics'] = [cache['cost_function']] + cache['metrics']
    cache.update({'W': {}, 'b': {}, 'VdW': {}, 'Vdb': {}, 'SdW': {}, 'Sdb': {}})
    np.random.seed(cache['random_seed'])
    L, init = cache['layers'], cache['init']
    for i in range(1, len(L)):
        initTypes = {'None': 1, 'He': 2 / L[i - 1], 'Xavier': 1 / L[i - 1], 'Benjamin': 6 / (L[i] + L[i - 1])}
        cache['W'][i] = np.random.uniform(-1, 1, (L[i], L[i - 1])) * np.sqrt(initTypes[init])
        cache['b'][i] = np.zeros((L[i], 1))
        cache['VdW'][i], cache['Vdb'][i], cache['SdW'][i], cache['Sdb'][i] = 0, 0, 0, 0
    return cache


def activation(z, i, derivative=False):
    ε = 0.00000001
    if i == 'sigmoid':
        if derivative is False:
            return 1 / (1 + np.exp(-z) +ε)
        else:
            return np.exp(-z) / np.power(1 + np.exp(-z) +ε, 2)
    elif i == 'relu':
        if derivative is False:
            return np.where(z > 0, z +ε, -0.01 * z +ε)
        else:
            return np.where(z > 0, 1 +ε, -0.01 +ε)
    elif i == 'linear':
        if derivative is False:
            return z
        else:
            return np.where(z == z, 1, 0)
    elif i == 'softmax':
        k = np.max(z, axis=0, keepdims=True)
        if derivative is False:
            return np.exp(z - k) / (np.sum(np.exp(z - k), axis=0, keepdims=True) +ε)
        # if derivative is False:
        #    exp = np.exp(z)
        #    return exp / np.sum(exp, axis = 0, keepdims = True)
        else:
            yhat = activation(z, i, derivative=False)
            dz = np.ones((yhat.shape[0], yhat.shape[1]))
            for i in range(y.shape[1]):
                dyhat_dz = (yhat[:, i] * (np.identity(yhat[:, i].shape[0]) - yhat[:, i])).T
                dz[:, i] = np.dot(dyhat_dz, (-1 / y.size) * (y / yhat)[:, i])
            return dz


def costFunction(y, yhat, costType='logloss', derivative=False):
    ε = 0.0000001
    if costType == 'logloss':
        if derivative == False:
            return np.mean(-(y * np.log(yhat +ε) + (1 - y) * np.log(1 - yhat +ε)))
        else:
            return -(1 / y.size) * (np.divide(y, yhat +ε) - np.divide(1 - y, 1 - yhat +ε))
    if costType == 'rmse':
        if derivative == False:
            return (1 / 2) * np.mean(np.power(y - yhat, 2))
        else:
            return (1 / y.size) * (yhat - y)
    if costType == 'softmax_logloss':
        if derivative == False:
            return np.mean(- y * np.log(yhat))
        else:
            return -(1 / y.size) * y / yhat


def evaluate(cacheTemp):
    if cacheTemp == None:
        return 'NA'
    else:
        yhat, y = cacheTemp['yhat'], cacheTemp['y']
        metrics, result = cacheTemp['metrics'], []
        result.append(round(costFunction(y, yhat, costType=cacheTemp['cost_function'], derivative=False), 4))
        if 'accuracy' in metrics:
            result.append(np.mean(np.round(np.mean(np.where(y == np.round(yhat), 1, 0)), 4)))
        if 'auc' in metrics:
            if (len(np.unique(y.ravel())) == 2):
                from sklearn.metrics import roc_auc_score
                result.append(round(roc_auc_score(y.ravel(), yhat.ravel()), 4))
            else:
                result.append(0)
        if 'R2' in metrics:
            from sklearn.metrics import r2_score
            result.append(round(r2_score(y.T, yhat.T), 4))
        return result


def score(x, y, cache):
    if y.size == 0:
        return None
    else:
        from copy import deepcopy
        cacheTemp = deepcopy(cache)
        cacheTemp['x'], cacheTemp['y'] = x, y
        cacheTemp = iteration(cacheTemp)
        return cacheTemp


def iteration(cache):
    x, y, L, m, costType = cache['x'], cache['y'], cache['layers'], cache['x'].shape[1], cache['cost_function']
    α, F, momentum, rmsprop, ε = cache['lrate'], cache['activations'], cache['momentum'], cache['rmsprop'], cache['epsilon']

    # ForwardProp
    A, Z = [x], [None]
    for i in range(1, len(L)):
        Z.append(np.dot(cache['W'][i], A[i - 1]) + cache['b'][i])
        A.append(activation(Z[i], F[i - 1], derivative=False))
    cache['yhat'] = A[-1]

    # BackProp
    dW, db = [], []
    for i in range(len(L) - 1, 0, -1):
        if i == len(L) - 1:
            dA = costFunction(y, A[-1], costType=costType, derivative=True)
            if F[i - 1] == 'softmax':
                dZ = activation(Z[i], F[i - 1], derivative=True)
            else:
                dZ = dA * activation(Z[i], F[i - 1], derivative=True)
        else:
            dA = np.dot(cache['W'][i + 1].T, dZ)
            dZ = dA * activation(Z[i], F[i - 1], derivative=True)
        dW.append(np.dot(dZ, A[i - 1].T) / m)
        db.append(dZ.mean(axis=1, keepdims=True))
    dW.append(None)
    dW.reverse()
    db.append(None)
    db.reverse()

    # Adam Optimizer
    for i in range(1, len(L)):
        cache['VdW'][i] = momentum * cache['VdW'][i] + (1 - momentum) * dW[i]
        cache['SdW'][i] = rmsprop * cache['SdW'][i] + (1 - rmsprop) * dW[i] ** 2
        cache['Vdb'][i] = momentum * cache['Vdb'][i] + (1 - momentum) * db[i]
        cache['Sdb'][i] = rmsprop * cache['Sdb'][i] + (1 - rmsprop) * db[i] ** 2
    for i in range(1, len(L)):
        cache['W'][i] = cache['W'][i] - α * cache['VdW'][i] / np.sqrt(cache['SdW'][i] + ε)
        cache['b'][i] = cache['b'][i] - α * cache['Vdb'][i] / np.sqrt(cache['Sdb'][i] + ε)
    return cache


def neuralNetwork(cache):
    cache = initialization(cache)
    print('NEURAL NETWORK')
    print(f"Training, Validation & Test Examples: {cache['X_train'].shape[1], cache['X_val'].shape[1], cache['X_test'].shape[1]}")
    print('Hyperparameters:')
    for i in ['cost_function', 'lrate', 'iterations', 'random_seed', 'layers', 'activations',
              'init', 'minibatchsize', 'momentum', 'rmsprop', 'epsilon', 'metrics']:
        print('\t', i, '-', cache[i])

    print('\nTraining Begins...\n')
    # Mini-Batch
    batch, mb, epoch = miniBatchGenerator(cache['X_train'].shape[1], cache['minibatchsize']), 0, 0
    for i in range(cache['iterations']):
        cache['x'] = cache['X_train'][:, batch[mb][0]:batch[mb][1]]
        cache['y'] = cache['y_train'][:, batch[mb][0]:batch[mb][1]]

        mb += 1
        if mb == len(batch):
            mb, epoch = 0, epoch + 1

        cache = iteration(cache)

        if (i % (cache['iterations'] / 10) == 0):
            print(f"Iter:{i}, Epoch:{epoch}, Train: {evaluate(score(cache['X_train'], cache['y_train'], cache))}, Eval: {evaluate(score(cache['X_val'], cache['y_val'], cache))}, Test: {evaluate(score(cache['X_test'], cache['y_test'], cache))}")
    print('\nTraining Concludes.\n')
    return cache


# REGRESSION PROBLEM
import pandas as pd
print('''\n%%% REGRESSION %%%\n
Task is to predict strength of concrete.
Source: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength \n''')
df = pd.read_csv('/Users/pranjal/Google Drive/github/neural_network/tests/concrete.csv')
y, x = np.array(df[['csMPa']]).T, np.array(df.drop('csMPa', axis=1)).T
x = featureScaling(x, scaleType='standardize', axis=1)
train, val, test = dataSplit(x, y, [80, 20, 0], shuffle=True)
params = {'X_train': train[0], 'y_train': train[1],
          'X_val': val[0], 'y_val': val[1],
          'X_test': test[0], 'y_test': test[1],
          'cost_function': 'rmse', 'lrate': 0.2, 'iterations': 1200, 'hidden_layers': [25],
          'activations': ['relu', 'linear'], 'minibatchsize': 824, 'metrics': ['R2']}
cache = neuralNetwork(params)
print('Comparision:')
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
model = LinearRegression()
model.fit(train[0].T, train[1].T)
print('Linear Reg R2: ', r2_score(val[1].T, model.predict(val[0].T)))
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
model = XGBRegressor(random_state=10)
model.fit(train[0].T, train[1].T)
print('XGB Regressor R2: ', r2_score(val[1].T, model.predict(val[0].T)))


# BINARY CLASSIFICATION PROBLEM
print('''\n%%%% Binary Classification %%%\n''')
from sklearn.datasets import make_gaussian_quantiles
import matplotlib.pyplot as plt
plt.title("Sample Dataset", fontsize='small')
x, y = make_gaussian_quantiles(n_features=2, n_classes=8, n_samples=10000)
y = np.where((y == 0) | (y == 2) | (y == 4) | (y == 6), 1, 0)
plt.scatter(x[:, 0], x[:, 1], marker='o', c=y, s=25, edgecolor='k')
plt.show()
x = featureScaling(x.T, scaleType='standardize', axis=1)
y = y.reshape(1, x.shape[1])
train, val, test = dataSplit(x, y, [80, 20, 0], shuffle=True)
params = {'X_train': train[0], 'y_train': train[1], 'X_val': val[0], 'y_val': val[1], 'X_test': test[0], 'y_test': test[1],
          'cost_function': 'logloss', 'lrate': 0.5, 'iterations': 5000, 'hidden_layers': [100, 50],
          'activations': ['relu', 'relu', 'sigmoid'], 'minibatchsize': 2048, 'metrics': ['accuracy', 'auc']}
cache = neuralNetwork(params)

print('Comparision:')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
model = LogisticRegression()
model.fit(train[0].T, train[1].T)
print('Logistic Regression AUC: ', roc_auc_score(val[1].T, model.predict_proba(val[0].T)[:, 1]))
from xgboost import XGBClassifier
from sklearn.metrics import r2_score
model = XGBClassifier(random_state=10)
model.fit(train[0].T, train[1].T)
print('XGBoost Classifier AUC: ', roc_auc_score(val[1].T, model.predict_proba(val[0].T)[:, 1]))


# MULTI-CLASSIFICATION TRAINING
print('''\n%%%% Multi-Class Classification %%%\n''')
from sklearn.datasets import load_iris
df = load_iris()
y, x = df.target.T.reshape(1, df.target.shape[0]), df.data.T
y = np.r_[np.where(y == 0, 1, 0), np.where(y == 1, 1, 0), np.where(y == 2, 1, 0)]
x = featureScaling(x, scaleType='standardize', axis=1)
params = {'X_train': x, 'y_train': y,
          'cost_function': 'softmax_logloss', 'lrate': 0.1, 'iterations': 50, 'hidden_layers': [5],
          'activations': ['relu', 'softmax'], 'minibatchsize': 150, 'metrics': ['accuracy']}
cache = neuralNetwork(params)
