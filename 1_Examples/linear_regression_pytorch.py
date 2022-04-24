'''
The classic linear regression model, this time in PyTorch and through OOPs. Comparision is made to the Sklearn module 
in terms on training R2, intercept and coefficients. Dataset on Concrete Strength is used from UCI ML Repository 
(https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength). Task is to predict/explain Concrete Strength
using Cement quantitiy, Blast furnace slag, fly ash, water and other attributes.

Sklearn's module uses the analytical solution so will always get to the solution faster and have better Training R2 than
Gradient Descent based methods. Three examples are presented: one with no interaction terms, one with only squared terms
and one with all interaction terms. In the first two cases, the analytical & GD solution is nearly identical; but in the
last case, the coefficients vary quite significantly because while GD gets us close to the optimal solution it does so
increasingly slowly.

And the difference between nearly optimal coefficients and optimal coefficients can be quite large (of the order 100x),
even though the Training R2 is not very different. My conjecture is that this difference will be eliminated when GD gets
even closer to optimal.

It is surprising, but GD based methods might be better that analytical based methods even from interpretation because
analytical solutions can give extreme values for coefficients. GD based solutions will take much longer to give
extreme values. Thus for stress testing, P-value, coefficient interpretation, GD based solutions might be
closer to the truth.

For example, say we are building a model for predicting credit risk as a function of 10-15 other variables including
interaction terms. Analytical solution might give 100x coefficient value for x1 as compared to GD based solutions.
When stress testing, if an outlier arrives in x1, the predicted value would be quite unrealistic. GD based solutions
present nearly optimal results, which can be more realistic and more robust to outliers.

'''

# Build Model
import numpy as np
import torch as t
import warnings
warnings.filterwarnings("ignore")

# Model Code


class LinearRegressionTorch:

    def __init__(s, y, x, λ=1, seed=42):
        print('Custom Linear Regression Model in Torch')
        print('State Initialized.')
        s.seed = t.manual_seed(seed)
        s.y = y
        s.x = x
        s.W = t.randn(x.shape[1], 1, requires_grad=True)
        s.b = t.randn(1, 1, requires_grad=True)
        s.m, s.n = x.shape

    def FeatureScale(s):
        print('Feature scaling complete.')
        μ = t.mean(s.x, axis=0, keepdims=True)
        σ = t.std(s.x, axis=0, keepdims=True)
        s.x = (x - μ)/σ

    def ForwardPass(s):
        s.yhat = t.mm(s.x, s.W) + s.b

    def CalculateLoss(s):
        s.rmse = t.mean(t.pow(s.y - s.yhat, 2)) + s.λ * t.mean(t.pow(s.W, 2))
        return s.rmse

    def BackProp(s):
        s.rmse.backward()
        s.dW = s.W.grad
        s.db = s.b.grad

    def GradientDescent(s, lrate=0.1, mom=0.9):
        s.mW = mom * s.mW - lrate * s.dW
        s.mb = mom * s.mb - lrate * s.db
        with t.no_grad():
            s.W = t.tensor(s.W + s.mW, requires_grad=True)
            s.b = t.tensor(s.b + s.mb, requires_grad=True)

    def Train(s, iterations=100, lrate=0.01, λ=1, mom=0.99, verbose=10):
        print('Training Begins...\n')
        s.mW = t.zeros(s.W.shape)
        s.mb = t.zeros(s.b.shape)
        s.λ = λ
        for i in range(iterations):
            s.ForwardPass()
            s.CalculateLoss()
            s.BackProp()
            s.GradientDescent(lrate, )

            if i % verbose == 0:
                print(f'Iter {i} - RMSE:{s.rmse.item(): 0.4}, R2:{s.R2(s.y, s.yhat).item():0.4}')
        print('\n...Training Ends.')

    def R2(s, y, yhat):
        RSS = t.sum(t.pow(y - yhat, 2))
        TSS = t.sum(t.pow(y - t.mean(y), 2))
        R2 = 1 - RSS / TSS
        return R2

    def Score(s, x, scale=True):
        print('Scored.\n')
        if scale == True:
            μ = t.mean(x, axis=0, keepdims=True)
            σ = t.std(x, axis=0, keepdims=True)
            x = (x - μ)/σ
        yhat = t.mm(x, s.W) + s.b
        return yhat


print('Example 1: Multivariate Regression')
# Load the data
PATH = '/Users/pranjal/Google Drive/Projects/github/neural_network/tests/concrete.csv'
df = np.loadtxt(PATH, skiprows=1, delimiter=',')
y, x = df[:, -1:], df[:, :-1]

# SKLEARN
x = (x - np.mean(x, 0, keepdims=True)) / np.std(x, axis=0, keepdims=True)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
LR = LinearRegression(normalize=False)
LR.fit(x, y)
print('Sklearn Linear Regression')
print(f'Train R2: {r2_score(y, LR.predict(x)):0.4}')
print(f'Intercept Parameter: ', LR.intercept_)
print(f'Slope Parameters: \n', LR.coef_.T, '\n')

# PYTORCH
y = t.tensor(df[:, -1:], dtype=t.float32)
x = t.tensor(df[:, :-1], dtype=t.float32)
model = LinearRegressionTorch(y, x)
model.FeatureScale()
model.Train(iterations=100, lrate=0.1, λ=0, verbose=10)
model.R2(model.y, model.yhat)
preds = model.Score(x)
print(f'Train R2: {model.R2(model.y, model.yhat)}')
print(f'Intercept Parameter: ', model.b.detach())
print(f'Slope Parameters: \n', model.W.detach())

print('Difference in coefficients')
print(np.round(LR.coef_.T - model.W.detach().numpy(), 2))


print('Example 2: Multi-Regression Model with Squared terms')
# Load the data
PATH = '/Users/pranjal/Google Drive/Projects/github/neural_network/tests/concrete.csv'
df = np.loadtxt(PATH, skiprows=1, delimiter=',')
y, x = df[:, -1:], df[:, :-1]
x = np.c_[x, df[:, :-1] * df[:, :-1]]

# SKLEARN
x = (x - np.mean(x, 0, keepdims=True)) / np.std(x, axis=0, keepdims=True)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
LR = LinearRegression(normalize=False)
LR.fit(x, y)
print('Sklearn Linear Regression')
print(f'Train R2: {r2_score(y, LR.predict(x)):0.4}')
print(f'Intercept Parameter: ', LR.intercept_)
print(f'Slope Parameters: \n', LR.coef_.T, '\n')


# PYTORCH
y = t.tensor(df[:, -1:], dtype=t.float32)
x = t.tensor(df[:, :-1], dtype=t.float32)
x = t.cat((x, x * x), 1)

model = LinearRegressionTorch(y, x)
model.FeatureScale()
model.Train(iterations=10000, lrate=0.1, λ=0, verbose=1000)
model.R2(model.y, model.yhat)
preds = model.Score(x)
print(f'Train R2: {model.R2(model.y, model.yhat)}')
print(f'Intercept Parameter: ', model.b.detach())
print(f'Slope Parameters: \n', model.W.detach())

print('Difference in coefficients')
print(np.round(LR.coef_.T - model.W.detach().numpy(), 2))


print('Example 3: Regression Model with all interaction terms.')
# Load the data
PATH = '/Users/pranjal/Google Drive/Projects/github/neural_network/tests/concrete.csv'
df = np.loadtxt(PATH, skiprows=1, delimiter=',')
y, x = df[:, -1:], df[:, :-1]

# Add square terms and interactions
for i in df[:, :-1].T:
    for j in df[:, :-1].T:
        temp = i * j
        x = np.c_[x, temp]

# SKLEARN
x = (x - np.mean(x, 0, keepdims=True)) / np.std(x, axis=0, keepdims=True)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
LR = LinearRegression(normalize=False)
LR.fit(x, y)
print('Sklearn Linear Regression')
print(f'Train R2: {r2_score(y, LR.predict(x)):0.4}')
print(f'Intercept Parameter: ', LR.intercept_)
print(f'Slope Parameters: \n', LR.coef_.T, '\n')

# PYTORCH
y = t.tensor(df[:, -1:], dtype=t.float32)
x = t.tensor(df[:, :-1], dtype=t.float32)

for i in df[:, :-1].T:
    for j in df[:, :-1].T:
        temp = t.tensor(i * j, dtype=t.float32)
        temp = temp.reshape(1030, 1)
        x = t.cat((x, temp), 1)

model = LinearRegressionTorch(y, x)
model.FeatureScale()
model.Train(iterations=100000, lrate=0.05, λ=0, verbose=10000)
model.R2(model.y, model.yhat)
preds = model.Score(x)
print(f'Train R2: {model.R2(model.y, model.yhat)}')
print(f'Intercept Parameter: ', model.b.detach())
print(f'Slope Parameters: \n', model.W.detach())

print('Difference in coefficients')
print(np.round(LR.coef_.T - model.W.detach().numpy(), 2))

# The End
