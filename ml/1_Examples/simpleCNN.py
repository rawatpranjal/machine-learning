''' This code builds a simple Convolutional Neural Network using Numpy & For-Loops in Python. 
A total of 360 8x8 Pixel Images of MNIST Handwritted Digits (labels - 0 and 1 only) are fed
into a two layer network, with one CONVOLUTION layer & one FULLY CONNECTED/(generic sigmoid) layer.

Architecture: Input-Image (8x8) -> Filtered-Image (6x6) -> Prediction (Yhat, 1x1)

We deny the second layer (FC/Dense) weights from being updated, in order to highlight the weight updates
being done in the CONV layer. 

Reference: 4th Module of Andrew Ng's Deep Learning Specialization. '''

# DATA UPLOAD & PRE-PROCESSING
print('Loading Handwritted Digits...')
import numpy as np
from sklearn.datasets import load_digits
digits = load_digits()
images = digits.data.reshape(-1, 8, 8)
labels = digits.target

print('Selecting only 1s and 0s...')
whereOnes = np.where(labels == 1, 1, 0)
images_ones = images[whereOnes==1]
whereZeros = np.where(labels == 0, 1, 0)
images_zeros = images[whereZeros==1]
INPUT = np.r_[images_ones.ravel(), images_zeros.ravel()].reshape(-1, 8, 8)
LABEL = np.r_[labels[whereOnes == 1], labels[whereZeros ==1]]

print('Shuffling...')
np.random.seed(1)
randomize = np.arange(len(INPUT))
np.random.shuffle(randomize)
INPUT = INPUT[randomize]
LABEL = LABEL[randomize]

print('Visualizing Labels & Raw Input...')
import matplotlib.pyplot as plt
img_no = 0
plt.imshow(INPUT[img_no], cmap="gray")
plt.title(label = f'Label = {LABEL[img_no]}')
plt.show()

img_no = 10
plt.imshow(INPUT[img_no], cmap="gray", label = 'Y = {LABEL[1]}')
plt.title(label = f'Label = {LABEL[img_no]}')
plt.show()

# BUILD CNN MODEL
def initCONV(image, f, scale=0.01):
    np.random.seed(1)
    input_height, input_width = image.shape
    W1 = np.random.normal(0, 1, (f, f)) * scale
    b1 = np.zeros((1, 1))
    output_height = int(input_height - f + 1)
    output_width = int(input_width - f + 1)
    Z1 = np.zeros((output_height, output_width))
    return W1, b1, Z1, f

def initFC(A1, scale=0.01):
    np.random.seed(1)
    Z2 = A1.ravel().reshape(-1, 1)
    n_prev = Z2.shape[0]
    W2 = np.random.normal(0, 1, (1, n_prev)) * scale
    b2 = np.zeros((1, 1))
    return W2, b2, Z2

def convolve(a, b):
    return np.sum(np.multiply(a, b))

def sigmoid(x):
    return (1 + np.exp(-x)) ** -1

def logloss(A2, y):
    return -np.mean(y*np.log(A2) + (1-y)*np.log(1-A2))

def forwardPassCONV(X, W1, b1, Z1, f):
    n_H, n_W = Z1.shape
    for h in range(n_H):
        for w in range(n_W):
            value = convolve(X[h:h+f, w:w+f], W1[:, :])
            Z1[h, w] = value + b1
    return Z1, Z1

def forwardPassFC(A1, W2, b2):
    A1 = A1.ravel().reshape(-1, 1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return A2, Z2

def backpropFC1(dZ2, A1):
    A1 = A1.ravel().reshape(-1, 1)
    dW2 = np.dot(A1, dZ2).T
    db2 = dZ2
    return dW2, db2

def backpropFC2(dZ2, W2, b2, A1):
    dA1_unroll = np.dot(W2.T, dZ2)
    dA1 = dA1_unroll.reshape(A1.shape)
    dZ1 = dA1 * (1-dA1)
    return dZ1

def backpropCONV(dZ1, X, f, W1, b1):
    (H, W) = dZ1.shape
    dW1 = np.zeros(W1.shape)
    db1 = np.zeros(b1.shape)
    for h in range(H):
        for w in range(W):
            dW1 += dZ1[h, w] * X[h:h+f, w:w+f]
            db1 += dZ1[h, w]
    return dW1, db1

def CNN(INPUT, LABEL, filterSize, epochs, learning_rate):
    # Initialize
    np.random.seed(2)
    W1, b1, Z1, f = initCONV(INPUT[0], f=filterSize)
    W2, b2, Z2 = initFC(Z1)
    m = len(INPUT)
    avg_epoch_loss, epoch_accuracy = [], []
    path_Z1, path_W1 = [], []

    for j in range(epochs):
        print(f'Epoch {j}...')
        loss, correct = 0, 0

        for i in range(m):
            X, y = INPUT[i], LABEL[i]

            # Forward Propagation
            A1, Z1 = forwardPassCONV(X, W1, b1, Z1, f)
            A2, Z2 = forwardPassFC(A1, W2, b2)

            # Evaluate
            print(f'\n Epoch {j}, {i}th Example:')
            print('Prediction:', round(A2[0][0], 2))
            print('True Label:', y)
            print('Loss Incurred:', round(logloss(A2, y), 2))
            loss += logloss(A2, y)
            pred = np.where(round(A2[0][0], 2)>0.5, 1, 0)
            correct += np.where(pred==y,1,0)

            # Backward Propagation
            dZ2 = A2 - y
            dW2, db2 = backpropFC1(dZ2, A1)
            dZ1 = backpropFC2(dZ2, W2, b2, A1)
            dW1, db1 = backpropCONV(dZ1, X, f, W1, b1)

            # Gradient Update (only on CONV Layer)
            #W2 -= lrate * dW2
            #b2 -= lrate * db2
            W1 -= learning_rate * dW1
            #b1 -= learning_rate * db1

        path_Z1.append(Z1)
        path_W1.append(W1)
        avg_epoch_loss.append(loss/m)
        epoch_accuracy.append(correct/m)
        
    return avg_epoch_loss, epoch_accuracy

# TRAIN & EVALUATE
avg_epoch_loss, epoch_accuracy = CNN(INPUT, LABEL, filterSize=3, epochs =10, learning_rate= 0.1)

print('Loss & Accuracy over Epochs')
plt.plot(avg_epoch_loss)
plt.title('Average Loss over Epoch')
plt.show()

plt.plot(epoch_accuracy)
plt.title('Average Accuracy over Epoch')
plt.show()
