'''
This code builds a simple Recurrent Neural Network for sequence/time series classification. 
A very simple sequence is fed into a 2 Layer RNN, to obtain predictions. 

Architecture:

y(t): (1x1) prediction at time t
a(t): (10x1) hidden activation at time t
x(t): (3x1) input at time t
  

          y(1)   y(2)          y(30)
           |      |             |
 a(0) -> a(1) -> a(2) -> ..... a(30)
           |      |             |
          x(1)   x(2)          x(30)
          
Since activations effect activations & predictions in the future, backpropagation through time involves calculating the 
gradients for every activation through same period and future predictions. The inspiration for this has been the 5th Module
of Andrew Ng's Deep Learning specialization on Coursera. 

'''

# Generate a Simple Sequence
import numpy as np

def random_sequence(possible_values, length, seed):
    np.random.seed(seed)
    x = []
    for i in range(length):
        n = np.random.choice(possible_values)
        x.append(n)
    return x

def as_some_function_of_(x):
    y = x.copy()
    for i in range(len(x)):
        if i == 0:
            y[i] = 0
        elif ((x[i] - x[i - 1]) == 1):
            y[i] = 1
        else:
            y[i] = 0
    return y

def oneHotEncode(x):
    rows, columns = len(np.unique(x)), x.shape[1]
    result = np.zeros((rows, columns), dtype='int')
    for i in range(columns):
        result[x[0][i], i] = 1
    return result

x = random_sequence([0, 1, 2], length=30, seed=2)
y = as_some_function_of_(x)
x, y = np.array(x).reshape(1, -1), np.array(y).reshape(1, -1)
X = oneHotEncode(x)

print('''\nSequence Logic:
x = 0 or 1 or 2 with equal probability
y = 1 if x(t)==x(t-1) + 1
  = 0 otherwise \n''')
print('Raw Input "x":', x, x.shape)
print('Output    "y":', y, y.shape)
print('One Hot Encoded "X": \n', X, X.shape)

# Build Model
def initParam(x, y, nodes, scale=0.01):
    np.random.seed(10)
    a = np.zeros((nodes, 1))
    Waa = np.random.normal(0, 1, (a.shape[0], a.shape[0])) * scale
    Wax = np.random.normal(0, 1, (a.shape[0], x.shape[0])) * scale
    Ba, By = np.zeros((nodes, 1)), np.zeros((1, 1))
    Way = np.random.normal(0, 1, (y[:, 0].shape[0], a.shape[0])) * scale
    params = (Waa, Wax, Way, Ba, By)
    return params

def sigmoid(x):
    return np.divide(1, 1 + np.exp(-x))

def logloss(y, yhat):
    return - np.mean(y * np.log(yhat) + (1 - y) * np.log((1 - yhat)))

def accuracy(y, yhat):
    return np.mean(np.where(np.where(yhat > 0.5, 1, 0) == y, 1, 0))

def feed_forward(X, y, params):
    (Waa, Wax, Way, Ba, By) = params
    T, yhat = X.shape[1], np.zeros(y.shape)
    a = np.zeros((Waa.shape[1], y.shape[1] + 1))
    for i in range(T):
        x_prev = X[:, [i]]
        a_prev = a[:, [i]]

        a_post = sigmoid(np.dot(Waa, a_prev) + np.dot(Wax, x_prev) + Ba)
        yhat_i = sigmoid(np.dot(Way, a_post) + By)

        a[:, [i + 1]] = a_post
        yhat[:, [i]] = yhat_i
    return yhat, a

def backprop(yhat, y, a, X, params):
    (Waa, Wax, Way, Ba, By) = params
    dWaa, dWax, dWay, dBa, dBy = 0 * Waa, 0 * Wax, 0 * Way, 0 * Ba, 0 * By

    T = y.shape[1]
    dZ = (1 / T) * (yhat - y)
    temp1, temp2 = 0, 0
    for i in range(T, 0, -1):
        dZ_t, a_t, x_t = dZ[:, [i - 1]], a[:, [i]], X[:, [i - 1]]

        # Top Layer Gradients
        dWay += np.dot(dZ_t, a_t.T)
        dBy += dZ_t

        # Recursive Gradients
        da_t = np.dot(dZ_t, Way).T + temp2
        temp1 = sigmoid(np.dot(Waa, a[:, [i - 1]]) + np.dot(Wax, X[:, [i - 1]]) + Ba)
        temp2 = temp1 * (1 - temp1) * np.dot(da_t.T, dWaa).T
        dz_t = da_t * (a_t * (1 - a_t))

        # Bottom Layer Gradients
        dWaa += np.dot(dz_t, a[:, [i - 1]].T)
        dWax += np.dot(dz_t, x_t.T)
        dBa += dz_t

    dparams = (dWaa, dWax, dWay, dBa, dBy)
    return dparams

def gradientUpdate(params, dparams, momentum, lrate=0.05):
    (mWaa, mWax, mWay, mBa, mBy) = momentum
    (Waa, Wax, Way, Ba, By) = params
    (dWaa, dWax, dWay, dBa, dBy) = dparams

    mWay = 0.9 * mWay + lrate * dWay
    mBy = 0.9 * mBy + lrate * dBy
    mWaa = 0.9 * mWaa + lrate * dWaa
    mWax = 0.9 * mWax + lrate * dWax
    mBa = 0.9 * mBa + lrate * dBa
    Way -= mWay
    By -= mBy
    Waa -= mWaa
    Wax -= mWax
    Ba -= mBa

    momentum = (mWaa, mWax, mWay, mBa, mBy)
    params = (Waa, Wax, Way, Ba, By)
    return params, momentum

# RNN
def RNN(X, y, hidden_layer_nodes, iterations, learning_rate):
    params = initParam(X, y, nodes=hidden_layer_nodes)

    momentum = (0, 0, 0, 0, 0)
    loss_path, accuracy_path = [], []

    for i in range(iterations):
        yhat, a = feed_forward(X, y, params)
        dparams = backprop(yhat, y, a, X, params)
        params, momentum = gradientUpdate(params, dparams,
                                          momentum, lrate=learning_rate)
        loss_path.append(logloss(y, yhat))
        accuracy_path.append(accuracy(y, yhat))

    return yhat, loss_path, accuracy_path, params


# Train Model
yhat, loss_path, accuracy_path, params = RNN(X, y, hidden_layer_nodes=10, iterations=1000, learning_rate=0.2)

# Evaluate on Training Set
print('Eval on Training Set')
print('Predicted:', np.where(yhat > 0.5, 1, 0))
print('Actual   :', y)

import matplotlib.pyplot as plt
print('Loss & Accuracy over Training Iterations')
plt.plot(loss_path)
plt.title('Loss over Training Iterations')
plt.show()

plt.plot(accuracy_path)
plt.title('Accuracy over Training Iterations')
plt.show()

# Evaluate on Test Set
print('Eval on new Test Example --  Ensure seed is different')
x = random_sequence([0, 1, 2], length=30, seed=41)
y = as_some_function_of_(x)
x, y = np.array(x).reshape(1, -1), np.array(y).reshape(1, -1)
X = oneHotEncode(x)
yhat, a = feed_forward(X, y, params)

print('Test "x"   :', x)
print('Test "y"   :', y)
print('Predictions:', np.where(yhat > 0.5, 1, 0))
print('Test Accuracy:', accuracy(y, yhat))
print('Test Logloss:', logloss(y, yhat))
