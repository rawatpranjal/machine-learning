'''
Core aspects of Andrew Ng's Machine Learning Coursera Course
Topic: Week 4 - Week 5: Neural Network/Multi-Layered Perceptron
Data: ex3data1.mat (from the Coursera Course)
Problem: Multi-Classification/Handwritten Digit Recognition
Features = 400 Greyscale Pixels
Observations = 5k
'''

PATH = '/Users/pranjal/Google Drive/python_projects/projects/courses/andrew_ng/machine_learning/algorithms/neural_network/df.txt'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score


def import_Xy(PATH):
    df = pd.read_csv(PATH, header=None)
    df = df.sample(frac=1)
    X = np.array(df.drop(0, axis=1))
    y_multi = np.array(df[0]).reshape(X.shape[0], 1)
    class_labels = list(set(list(df[0])))
    for i in class_labels:
        var = str(i)
        df[var] = 0
        df.loc[df[0] == i, var] = 1
    y = np.array(df.iloc[:, -len(class_labels):])
    return X, y


def featureScale(X):
    temp0 = X - X.mean(axis=0)
    temp1 = X.max(axis=0) - X.min(axis=0)
    return np.divide(temp0, temp1 + 0.00000001)


def train_test_split(X, y, size):
    ''' X_train, X_test, y_train, y_test '''
    df = np.hstack((y, X))
    np.random.shuffle(df)
    X, y = df[:, y.shape[1]:], df[:, :y.shape[1]]
    m = round(y.shape[0] * size)
    return X[m:, :], X[0:m, :], y[m:, :], y[0:m, :]


def theta_structure(X, y, hidden_layers):
    layers = [X.shape[1]]
    layers.extend(hidden_layers)
    layers.append(y.shape[1])
    L_no = len(layers)
    theta = []
    for i in range(0, L_no - 1):
        row = layers[i + 1]
        col = layers[i] + 1
        theta_temp = np.random.normal(0, 1, (row, col))
        theta.append(theta_temp)
    return theta


def sigmoid(z):
    '''Element wise Logistic Transformation on Array'''
    sigmoid = (1 + np.e ** (-z)) ** (-1)
    return np.round(sigmoid, 4)


def logLoss(h, y):
    '''Log Loss Cost'''
    m = y.shape[0]
    n = y.shape[1]
    loss = y * np.log(h + 0.0000000001)
    loss += (1 - y) * np.log(1 - h + 0.0000000001)
    loss = sum(sum(loss))
    loss = loss / (m * n)
    return np.round(-loss, 16)


def multiClassAccuracy(h, y):
    '''Mutli Class Accuracy (Unweighted)'''
    y_multi = np.argmax(y[:, :], axis=1).reshape(y.shape[0], 1) + 1
    y_multi_pred = np.argmax(h[:, :], axis=1).reshape(h.shape[0], 1) + 1
    diff = y_multi - y_multi_pred
    accuracy = np.where(diff == 0, 1, 0)
    return sum(sum(accuracy)) / len(accuracy)


def squaredCost(h, y):
    '''Squared Error Cost'''
    m = y.shape[0]
    n = y.shape[1]
    loss = 0.5 * (h - y) ** 2
    loss = sum(sum(loss))
    loss = loss / (m * n)
    return np.round(loss, 16)


def feed_forward(theta, X):
    '''Input: theta, ith X values; Output: all Outputs'''
    vecInput = np.c_[np.ones(X.shape[0]), X]
    output = [vecInput]
    cnt = 1
    # print(f'Layer {cnt} Output (Raw): ')
    # print(vecInput.shape)
    for i in theta[:-1]:
        vecInput = np.dot(vecInput, i.T)
        vecInput = sigmoid(vecInput)
        vecInput = np.c_[np.ones(vecInput.shape[0]), vecInput]
        output.append(vecInput)
        cnt += 1
        # print(f'\nLayer {cnt} Output (Hidden): ')
        # print(vecInput.shape)
    vecInput = np.dot(vecInput, theta[-1].T)
    vecInput = sigmoid(vecInput)
    output.append(vecInput)
    cnt += 1
    # print(f'\n Layer {cnt} Output (Final): ')
    # print(vecInput.shape)
    return output


def back_propagation(theta, output, y_example):
    '''Input: theta, outputs, ith Y example; Output - Deltas'''
    deltas = []
    cnt = len(output)
    delta = output[-1] - y_example
    # print(f'Layer {cnt} Delta: ')
    # print(delta.shape)
    deltas.append(delta)
    cnt -= 1
    temp = output.copy()
    temp.reverse()
    for i in temp[1:-1]:
        # print(f'\nLayer {cnt} Delta: ')
        cnt -= 1
        sigmoid_diff = np.multiply(i, 1 - i)
        delta = np.dot(delta, theta[cnt])
        delta = np.multiply(delta, sigmoid_diff)
        deltas.append(delta)
        # print(delta.shape)
        delta = delta[:, 1:]
    deltas.reverse()
    return deltas


def gradient_update(theta, output, deltas, mom, α):
    no_layers = len(theta)
    α = 0.01
    momentum = [0 for i in theta]
    for i in range(no_layers):
        # print(f'Theta in layer {i + 1}: ')
        if i == no_layers - 1:
            momentum[i] = mom * momentum[i] - α * np.dot(deltas[i].T, output[i])
            theta[i] += momentum[i]
        else:
            momentum[i] = mom * momentum[i] - α * np.dot(deltas[i][:, 1:].T, output[i])
            theta[i] += momentum[i]
        # print(theta[i].shape)
    return theta


def learning_curve(train, val):
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.plot([i[0] for i in train], label='Training LogLoss')
    plt.plot([i[0] for i in val], label='Evaluation LogLoss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot([i[1] for i in train], label='Training Accuracy')
    plt.plot([i[1] for i in val], label='Evaluation Accuracy')
    plt.legend()
    plt.show()


def neural_net(theta, X, y, eval_set, epoch_cnt, α, mom):
    m = len(y)
    train = []
    val = []
    for i in range(epoch_cnt):
        print(f'XXXXXXXXXXXXXXXXXXXXXXXXXXXX  BEGINNING EPOCH {i + 1} XXXXXXXXXXXXXXXXXXXX')
        # Training
        output = feed_forward(theta, X)
        deltas = back_propagation(theta, output, y)
        theta = gradient_update(theta, output, deltas, α, mom)

        # Cost & Accuracy
        cost = logLoss(output[-1], y)
        accuracy = multiClassAccuracy(output[-1], y)
        train.append([cost, accuracy])
        print(f'TRAIN LOGLOSS IN THIS EPOCH: {cost:.4f}')
        print(f'TRAIN ACCURACY IN THIS EPOCH: {accuracy:.4f}')

        # Val Cost & Accuracy
        output_val = feed_forward(theta, eval_set[0])
        cost = logLoss(output_val[-1], eval_set[1])
        accuracy = multiClassAccuracy(output_val[-1], eval_set[1])
        val.append([cost, accuracy])
        print(f'VAL LOGLOSS IN THIS EPOCH: {cost:.4f}')
        print(f'VAL ACCURACY IN THIS EPOCH: {accuracy:.4f} \n')
    return theta, train, val



# Traing & Evaluate Neural Network
np.random.seed(10)
X, y = import_Xy(PATH)
print('X shape: ', X.shape, 'y shape: ', y.shape)
X = featureScale(X)
print('Feature Scaling Done!')
α = 0.1
mom = 0.9
epoch_cnt = 200
hidden_layers = [500]
theta = theta_structure(X, y, hidden_layers)
print("Network Architecture: ", X.shape, [i.shape for i in theta], y.shape)
X_train, X_val, y_train, y_val = train_test_split(X, y, 0.2)
theta, train, val = neural_net(theta, X_train, y_train, [X_val, y_val], epoch_cnt, α, mom)
learning_curve(train, val)
