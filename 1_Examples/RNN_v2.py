'''
Building RNNs with PyTorch Modules and estimating patterns in simple autoregressive sequences. 
'''

#############################################################
## EXAMPLE 1: Simple Dependent Binary Sequences
## Generate X[t] a (3,1) vector over time periods T. Here T = 200.  
## When X[t] - X[t-1] in [-1, 1], then Y[t] = 1 else 0.
#############################################################

# Generate Sequence
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
        elif (x[i] - x[i - 1] == 1) or (x[i] - x[i - 1] == -1) or (x[i] == x[i - 1]):
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

x = random_sequence([0, 1, 2], length=200, seed=2)
y = as_some_function_of_(x)
x, y = np.array(x).reshape(1, -1), np.array(y).reshape(1, -1)
X = oneHotEncode(x)

import torch as t
x = t.tensor(X, dtype = t.float)
y = t.tensor(y, dtype = t.float)

print('\nTrain Sequence')
print(x[:, 0:15])
print(y[:, 0:15])

# RNN, Train, Score Functions
import torch.nn as nn
class RNN(nn.Module):
    def __init__(self, n_X, n_H, n_Y):
        super(RNN, self).__init__()
        self.H = t.randn(n_H)
        self.i2h = nn.Linear(n_X + n_H, n_H)
        self.i2o = nn.Linear(n_H, n_Y)
        self.f1 = nn.Tanh()
        self.f2 = nn.Sigmoid()

    def forward(self, x, h):
        x = t.cat((x, h))
        h = self.f1(self.i2h(x))
        yhat = self.f2(self.i2o(h))
        return yhat, h
      
def train(x, y, rnn, epochs = 50):
    T = x.shape[1] - 1
    optimizer = t.optim.Adam(rnn.parameters(), lr=0.1)
    loss_fn = t.nn.BCELoss()
    optimizer.zero_grad()

    for epoch in range(epochs):
        loss = 0
        for time in range(T):
            yhat, rnn.H = rnn.forward(x[:, time], rnn.H)
            loss += loss_fn(yhat, y[:, time])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
def score(x, rnn):
    T = x.shape[1]
    YHAT = t.zeros(1, T)
    YHAT[:, 0], h = rnn.forward(x[:, 0], rnn.H)
    for i in range(1, T):
        YHAT[:, i], h = rnn.forward(x[:, i], h)
    return YHAT

# Train on Sequence
rnn = RNN(n_X = 3, n_H = 12, n_Y = 1)
train(x, y, rnn)
print('Train Accuracy: ', t.sum(t.round(score(x, rnn))==y).float()/y.shape[1])

# Test a new Sequence
x = random_sequence([0, 1, 2], length=200, seed=10)
y = as_some_function_of_(x)
x, y = np.array(x).reshape(1, -1), np.array(y).reshape(1, -1)
X = oneHotEncode(x)
import torch as t
x = t.tensor(X, dtype = t.float)
y = t.tensor(y, dtype = t.float)
print('\nTest Sequence')
print(x[:, 0:15])
print(y[:, 0:15])
print(t.round(score(x, rnn))[:, 0:15])
print('Test Accuracy: ', t.sum(t.round(score(x, rnn))==y).float()/y.shape[1])

#############################################################
## EXAMPLE 2: AR(1) Processes
## Generate Y[t] = 0.5 * Y[t-1] + e[t] 
## Estimate with RNN.
#############################################################

# Create ARMA Process
import torch as t
t.manual_seed(3)
N = 200
e = t.randn(1, N) 
y = t.zeros(1, N)
for i in range(N-1):
    y[:, i+1] = 0 + 0.5 * y[:, i] + e[:, i]

x = y[:, 0:N-1]
y = y[:, 1:N]

print('\nTrain Sequence')
print(x[:, 0:15])
print(y[:, 0:15])

# Build RNN, Train & Score
import torch.nn as nn
class RNN(nn.Module):
    def __init__(self, n_X, n_H, n_Y):
        super(RNN, self).__init__()
        self.H = t.randn(n_H)
        self.i2h = nn.Linear(n_X + n_H, n_H)
        self.i2o = nn.Linear(n_H, n_Y)
        self.f = nn.Tanh()
    def forward(self, x, h):
        x = t.cat((x, h))
        h = self.f(self.i2h(x))
        yhat = self.i2o(h)
        return yhat, h

def train(x, y, rnn, epochs = 50):
    T = x.shape[1] - 1
    optimizer = t.optim.Adam(rnn.parameters(), lr=0.1)
    loss_fn = t.nn.MSELoss()
    optimizer.zero_grad()

    for epoch in range(epochs):
        loss = 0
        for time in range(T):
            yhat, rnn.H = rnn.forward(x[:, time], rnn.H)
            loss += loss_fn(yhat, y[:, time])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
def score(x, rnn):
    T = x.shape[1]
    YHAT = t.zeros(1, T)
    YHAT[:, 0], h = rnn.forward(x[:, 0], rnn.H)
    for i in range(1, T):
        YHAT[:, i], h = rnn.forward(x[:, i], h)
    return YHAT
  
# Train the Model, Eval with RMSE
rnn = RNN(n_X = 1, n_H = 1, n_Y = 1)
train(x, y, rnn)
loss_fn = t.nn.MSELoss()
print('Train RMSE: RNN Pred, Y[t-1] as Pred, Y[t] as Pred')
loss_fn(score(x, rnn), y).item(), loss_fn(x, y).item(), loss_fn(y, y).item()

# Test a New Sequence
import torch as t
t.manual_seed(100)
N = 200
e = t.randn(1, N) 
y = t.zeros(1, N)
for i in range(N-1):
    y[:, i+1] = 0 + 0.5 * y[:, i] + e[:, i]

x = y[:, 0:N-1]
y = y[:, 1:N]

print(x[:, 0:15])
print(y[:, 0:15])

print('Test RMSE: RNN Pred, Y[t-1] as Pred, Y[t] as Pred')
loss_fn(score(x, rnn), y).item(), loss_fn(x, y).item(), loss_fn(y, y).item()
