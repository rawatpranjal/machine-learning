test_path = '/axp/rim/imsadsml/dev/ppranja/kaggle/mnist/test.csv.filepart'
train_path = '/axp/rim/imsadsml/dev/ppranja/kaggle/mnist/train.csv.filepart'
sub_path = '/axp/rim/imsadsml/dev/ppranja/kaggle/mnist/sample_submission.csv.filepart'
new_sub_path = '/axp/rim/imsadsml/dev/ppranja/kaggle/mnist/submission.csv'

import pandas as pd, numpy as np
sub = pd.read_csv(sub_path)
test = pd.read_csv(test_path)
train = pd.read_csv(train_path)

temp = train.drop('label', axis = 1)
y, x = np.array(train[['label']]), temp/255
x = np.array(x)
x = x.reshape(42000, 1, 28, 28)

import torch as t
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
y_train = t.tensor(y_train, dtype = t.long)
y_train = y_train.reshape(-1)
y_test = t.tensor(y_test, dtype = t.long)
y_test = y_test.reshape(-1)

x_train = t.tensor(x_train, dtype = t.float)
x_test = t.tensor(x_test, dtype = t.float)

from torch import nn
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 16)
    
# Model Architechture
n_input, n_output = x_train.shape[1], 1
model = t.nn.Sequential(
t.nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 7, stride = 1),
t.nn.AvgPool2d(2),
t.nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 4, stride = 1),
t.nn.AvgPool2d(2),
t.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 4, stride = 1),
Flatten(),
t.nn.Linear(16, 10), 
t.nn.Softmax(dim = 1)
)

model(x_train).shape
model(x_train).argmax(dim = 1)

loss_fn = t.nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.1)

# Eval Metric
from sklearn.metrics import accuracy_score
def eval_metric(a, b):
    pred = a.argmax(dim = 1)
    return round(accuracy_score(pred.detach().numpy(), b.detach().numpy()), 2)

# Train
for i in range(100):
    loss = loss_fn(model(x_train), y_train)
    if i % 10 == 0:
        print(i, round(loss.item(), 2), eval_metric(model(x_train), y_train), eval_metric(model(x_test), y_test))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

test = np.array(test).reshape(28000, 1, 28, 28)
pred = model(t.tensor(test, dtype = t.float))
pred = pred.argmax(dim = 1)
sub['Label'] = pd.DataFrame(pred.detach().numpy())
sub.to_csv(new_sub_path, index=False)
