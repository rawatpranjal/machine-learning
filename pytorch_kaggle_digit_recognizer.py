#############################################################
## Image Recognition on MNIST Dataset
## Public Leader board score -> 99.37%, Rank at that time 568/2274
## Data: https://www.kaggle.com/c/digit-recognizer/data
## Run on Colab, reduce the learning rate over time
#############################################################

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Load Data
PATH = "/content/drive/My Drive/Github Projects/Digit Recognizer/"
test_path = PATH + 'test.csv'
train_path = PATH + 'train.csv'
sub_path = PATH + 'sample_submission.csv'
new_sub_path = PATH + 'submission.csv'

import pandas as pd, numpy as np
sub = pd.read_csv(sub_path)
train = pd.read_csv(train_path)

# Normalize & reshape Images
temp = train.drop('label', axis = 1)
y, x = np.array(train[['label']]).reshape(-1), temp/255
x = np.array(x)
x = x.reshape(42000, 1, 28, 28)

# Visualise Images & Labels
import matplotlib.pyplot as plt
for i in range(8):
    plt.imshow(x[i].squeeze(), cmap=plt.get_cmap('gray'))
    plt.title(y[i].item())
    plt.show()

######################################
## Enable GPU, iterable PyTorch Datasets
######################################

# Load PyTorch
import torch as t
t.manual_seed(42)
gpu = 'cuda' if t.cuda.is_available() else 'cpu'
cpu = 'cpu'

# Train Test Split
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Create Datasets
from torch.utils.data import TensorDataset
x_train = t.from_numpy(x_train).float().to(gpu)
y_train = t.from_numpy(y_train).to(gpu).long()
x_val = t.from_numpy(x_val).float().to(gpu)
y_val = t.from_numpy(y_val).to(gpu).long()
train_data = TensorDataset(x_train, y_train)
val_data = TensorDataset(x_val, y_val)

# Iterable Batches
from torch.utils.data import DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=4096, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=4096, shuffle=True)


#########################
## Network Architechture
#########################

# Model Architechture
class Flatten(t.nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)

model = t.nn.Sequential(
    t.nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3),
    t.nn.BatchNorm2d(32),
    t.nn.LeakyReLU(),
    
    t.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 7),
    t.nn.BatchNorm2d(64),
    t.nn.LeakyReLU(),
    t.nn.MaxPool2d(2, 2),
    
    t.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 9),
    t.nn.BatchNorm2d(128),
    t.nn.LeakyReLU(), 
    t.nn.MaxPool2d(2, 2),
    Flatten(),

    t.nn.Linear(128, 64),
    t.nn.BatchNorm1d(64),
    t.nn.LeakyReLU(),

    t.nn.Linear(64, 32),
    t.nn.BatchNorm1d(32),
    t.nn.LeakyReLU(),
    
   t.nn.Linear(32, 10),
   t.nn.Softmax(dim = 1),
)
model.cuda()
print(model)

# Loss Function & Optimizer
loss_fn = t.nn.CrossEntropyLoss()
optimizer = t.optim.Adam(model.parameters(), lr=0.01, weight_decay = 0.01)

# Evaluation Metric
from sklearn.metrics import accuracy_score
def eval_metric(a, b):
  pred = a.argmax(dim = 1)
  acc = t.mean(t.where(pred == b, t.ones(b.shape).cuda(), t.zeros(b.shape).cuda()))
  return round(acc.item(), 4)

######################
## Train the Network
######################

n_epochs = 10
for epoch in range(n_epochs):
  cnt = 1
  for x_batch, y_batch in train_loader:
      yhat = model(x_batch)
      loss = loss_fn(yhat, y_batch)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      cnt += 1
      
      x_val, y_val = next(iter(val_loader))
      print(f'Epoch: {epoch + 1}, Iteration: {cnt}, Loss: {round(loss.item(), 4)}, Train Acc: {eval_metric(model(x_batch), y_batch)}, Test Acc: {eval_metric(model(x_val), y_val)}')
   

######################
## Score Test Data
######################

# Preprocess Test Images 
test = pd.read_csv(test_path)
test = np.array(test).reshape(28000, 1, 28, 28)
test = t.tensor(test, dtype = t.float)/255.0
test = test.to(gpu)
test_data = TensorDataset(test, t.ones(test.shape[0]).cuda())
test_loader = DataLoader(dataset=test_data, batch_size=4096, shuffle=False)

# Check predictions on Test Images
sample = test[2500:3500]
sample_preds = model(sample).argmax(dim = 1)[0:10]
import matplotlib.pyplot as plt
for i in range(9):
    plt.imshow(sample[i].cpu().squeeze(), cmap=plt.get_cmap('gray'))
    plt.title(sample_preds[i].item())
    plt.show()
    
# Score in Batches
ypred_whole = np.array([], dtype = 'int')
for i, _ in test_loader:
    ypred = model(i)
    ypred = ypred.argmax(dim = 1)  
    ypred = ypred.detach().cpu().numpy()
    ypred_whole = np.r_[ypred_whole, ypred]
    
# Save Submission
sub['Label'] = ypred_whole
sub.to_csv(new_sub_path, index=False)
