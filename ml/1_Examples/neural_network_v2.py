'''
Derived from Andrew Ng's Deep Learning Specialization (Coursera)
Course 1: Neural Networks and Deep Learning
Problem: Binary-Classification on Balanced Datasets
Datasets: Toy Datasets & Synthetic

Summary: This version of the classic deep Neural Network contains
support for alternative activation functions like ReLU, TanH and Linear. 
We can plot error curves for train-validation-test datasets to observe
if training is proceeding correctly. Results are benchmarked to popular 
packages like LogisticRegression, XGBoost, Sklearn MLP.
'''

import numpy as np
np.random.seed(4994)
import warnings
warnings.filterwarnings("ignore")

'''
NEURAL NETWORK
'''


def initLayers(L, X_train, randomState):
    scaleW = 1 / len(L)
    np.random.seed(randomState)
    n = X_train.shape[1]
    A, W, B = [X_train], [], []
    for i in range(0, len(L) - 1):
        A.append(np.random.normal(0, 1, (L[i + 1], n)))
        W.append(np.random.normal(0, 1, (L[i + 1], L[i])) * scaleW)
        B.append(np.random.normal(0, 1, (L[i + 1], 1)) * scaleW)
    return A, W, B


def logloss(y, yhat):
    ε = 0.00001
    return - np.mean(y * np.log(yhat + ε) + (1 - y) * np.log(1 - yhat + ε))


def activationForward(z, i, activations):
    ε = 0.00001
    if activations[i] == "relu":
        return np.where(z >= 0, z + ε, 0.01 * z + ε)
    if activations[i] == "sigmoid":
        return (1 + np.exp(-z + ε)) ** -1
    if activations[i] == "tanh":
        return np.divide(np.exp(z + ε) - np.exp(-z + ε), np.exp(z + ε) + np.exp(-z + ε))
    if activations[i] == "linear":
        return z + ε


def forwardProp(A, W, B, L, activations):
    for i in range(len(L) - 1):
        Ztemp = np.dot(W[i], A[i]) + B[i]
        A[i + 1] = activationForward(Ztemp, i, activations)
    return A


def activationBP(z, i):
    ε = 0.00001
    if activations[i] == "sigmoid":
        return z * (1 - z) + ε
    if activations[i] == "relu":
        return np.where(z >= 0, 1, 0.01)
    if activations[i] == "tanh":
        return 1 - np.power(z + ε, 2)
    if activations[i] == "linear":
        return 1


def backProp(A, W, B, L, y):
    dW, dB, dZ = W.copy(), B.copy(), A.copy()
    m = A[0].shape[1]
    for l in range(len(L) - 1, 0, -1):
        if l == len(L) - 1:
            dZ[l] = ((1 - y) / (1 - A[l]) - y / A[l]) * activationBP(A[l], l - 1)
        else:
            dZ[l] = np.dot(W[l].T, dZ[l + 1]) * activationBP(A[l], l - 1)
        dW[l - 1] = (1 / m) * np.dot(dZ[l], A[l - 1].T)
        dB[l - 1] = (1 / m) * np.sum(dZ[l], axis=1, keepdims=True)
    return dW, dB


def gradientDescent(dW, dB, W, B, L, α=0.01, momentum=0.9):
    deltaW, deltaB = [0 for i in dW], [0 for i in dB]
    for i in range(len(L) - 1):
        deltaW[i] = momentum * deltaW[i] + α * dW[i]
        W[i] -= deltaW[i]
        B[i] -= deltaB[i]
    return W, B


def neuralNetwork(L, activations, learning_rate, epochs, X_train, y_train, val_set, test_set):
    print('\n NEURAL NETWORK \n')
    A, W, B = initLayers(L, X_train, randomState=4)
    A_val, temp, temp = initLayers(L, val_set[0], randomState=4)
    A_test, temp, temp = initLayers(L, test_set[0], randomState=4)
    trainLoss, valLoss, testLoss = [], [], []
    for i in range(epochs):
        A = forwardProp(A, W, B, L, activations)
        A_val = forwardProp(A_val, W, B, L, activations)
        A_test = forwardProp(A_test, W, B, L, activations)

        trainLoss.append(round(logloss(y_train, A[-1]), 2))
        valLoss.append(round(logloss(val_set[1], A_val[-1]), 2))
        testLoss.append(round(logloss(test_set[1], A_test[-1]), 2))

        dW, dB = backProp(A, W, B, L, y_train)
        W, B = gradientDescent(dW, dB, W, B, L, α=learning_rate)
        if i % round(epochs / 10) == 0:
            print(f'Loss in Epoch {i}: Train - {trainLoss[-1]}, Val - {valLoss[-1]}, Test {testLoss[-1]}')
    return W, B, A[-1], A_val[-1], A_test[-1], trainLoss, valLoss, testLoss


'''
PREPROCESSING & MODEL EVALUATION
'''


def featureScaling(X):
    ε = 0.00001
    meanDeviation = X - np.mean(X, axis=1, keepdims=True)
    minMaxRange = np.max(X, axis=1, keepdims=True) + ε
    minMaxRange -= np.min(X, axis=1, keepdims=True)
    proportion = meanDeviation / minMaxRange
    return proportion


def dataSplit(X, y, split, shuffle=True):
    a, b, c = split[0] / sum(split), split[1] / sum(split), split[2] / sum(split)
    if shuffle == True:
        df = np.r_[y, X]
        df = df.T
        np.random.shuffle(df)
        df = df.T
    y, X = df[0, :].reshape(1, X.shape[1]), df[1:, :]
    cut1, cut2 = round(X.shape[1] * a), round(X.shape[1] * b)
    return X[:, 0:cut1], X[:, cut1:cut1 + cut2], X[:, cut1 + cut2:], y[:, 0:cut1], y[:, cut1:cut1 + cut2], y[:, cut1 + cut2:]


def accuracy(y, ypred):
    return np.mean(np.where(np.round(ypred) == y, 1, 0))


def errorCurves(trainLoss, valLoss, testLoss):
    import matplotlib.pyplot as plt
    plt.plot(trainLoss, label='Training Error', c='g')
    plt.plot(valLoss, label='Validation Error', c='b')
    plt.plot(testLoss, label='Test Error', c='r')
    plt.title('Error Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Logloss')
    plt.legend()
    plt.show()


def modelEvaluation(y, X, L, activations, α, epochs, split=[60, 20, 20]):
    X = featureScaling(X)
    X_train, X_val, X_test, y_train, y_val, y_test = dataSplit(X, y, split, shuffle=True)
    print(f'Examples: {m}, Features: {n}, Target Classes: {r}')
    print(f'Train-Val-Test split: {split}')
    print(f'Hyperparameters: \n\tNetwork Units - {L}, \n\tActivations - {activations} \n\tLearning Rate - {α} \n\tEpochs - {epochs}')
    W, B, trainPred, valPred, testPred, trainLoss, valLoss, testLoss = neuralNetwork(L, activations, α, epochs, X_train, y_train, 
                                                                                     val_set=[X_val, y_val], test_set=[X_test, y_test])
    print(f'\n ACCURACY - Train:{accuracy(y_train, trainPred)}, Val: {accuracy(y_val, valPred)}, Test: {accuracy(y_test, testPred)} ')
    errorCurves(trainLoss, valLoss, testLoss)
    benchmark(X_train, X_test, y_train, y_test, testPred)


'''
BENCHMARKING
'''


def benchmarkMLP(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier()
    model.fit(X_train.T, y_train.T)
    y_pred = model.predict(X_test.T)
    return accuracy(y_test, y_pred)


def benchmarkXGB(X_train, X_test, y_train, y_test):
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.fit(X_train.T, y_train.T)
    y_pred = model.predict(X_test.T)
    return accuracy(y_test, y_pred)


def benchmarkLogistic(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    model.fit(X_train.T, y_train.T)
    y_pred = model.predict(X_test.T)
    return accuracy(y_test, y_pred)


def benchmarkRandom(X_train, X_test, y_train, y_test):
    eventRate = np.mean(y_train)
    testExamples = y_test.shape[1]
    y_pred = np.random.binomial(1, eventRate, size=(1, testExamples))
    return accuracy(y_test, y_pred)


def benchmark(X_train, X_test, y_train, y_test, testPred):
    import matplotlib.pyplot as plt
    plt.scatter(1, benchmarkRandom(X_train, X_test, y_train, y_test), label='Random')
    plt.scatter(2, benchmarkLogistic(X_train, X_test, y_train, y_test), label='Logistic')
    plt.scatter(3, benchmarkMLP(X_train, X_test, y_train, y_test), label='Sklearn MLP')
    plt.scatter(4, benchmarkXGB(X_train, X_test, y_train, y_test), label='XGBoost')
    plt.scatter(5, accuracy(y_test, testPred), label='NN MODEL')
    plt.title('Benchmarking')
    plt.xlabel('Models')
    plt.ylabel('Accuracy Score')
    plt.legend()
    plt.show()


'''
EVALUATIONS
'''

print('\n%%%%%%%%%%% EVALUATION 1 %%%%%%%%%%%\n')
from sklearn.datasets import load_iris
df = load_iris()
y, X = df.target.T.reshape(1, df.target.shape[0]), df.data.T
y = np.where(y == 0, 1, 0)
m, n, r = X.shape[1], X.shape[0], y.shape[0]
L = [n, 10, r]
activations = ['linear', 'sigmoid']
α, epochs = 0.1, 5000
print(modelEvaluation(y, X, L, activations, α, epochs, split=[60, 20, 20]))


print('\n%%%%%%%%%%% EVALUATION 2 %%%%%%%%%%%\n')
from sklearn.datasets import load_iris
df = load_iris()
y, X = df.target.T.reshape(1, df.target.shape[0]), df.data.T
y = np.where(y == 0, 1, 0)
m, n, r = X.shape[1], X.shape[0], y.shape[0]
L = [n, 10, r]
activations = ['tanh', 'sigmoid']
α, epochs = 0.1, 5000
print(modelEvaluation(y, X, L, activations, α, epochs, split=[60, 20, 20]))


print('\n%%%%%%%%%%% EVALUATION 3 %%%%%%%%%%%\n')
from sklearn.datasets import load_digits
df = load_digits()
y, X = df.target.T.reshape(1, df.target.shape[0]), df.data.T
y = np.where(y == 0, 1, 0)
m, n, r = X.shape[1], X.shape[0], y.shape[0]
L = [n, 10, r]
activations = ['sigmoid', 'sigmoid']
α, epochs = 0.1, 5000
print(modelEvaluation(y, X, L, activations, α, epochs, split=[60, 20, 20]))


print('\n%%%%%%%%%%% EVALUATION 4 %%%%%%%%%%%\n')
from sklearn.datasets import make_moons
df = make_moons(n_samples=20000, noise=.05)
y, X = df[1].T.reshape(1, df[0].shape[0]), df[0].T
y = np.where(y == 0, 1, 0)
m, n, r = X.shape[1], X.shape[0], y.shape[0]
L = [n, 10, r]
activations = ['relu', 'sigmoid']
α, epochs = 0.5, 10000
print(modelEvaluation(y, X, L, activations, α, epochs, split=[60, 20, 20]))
