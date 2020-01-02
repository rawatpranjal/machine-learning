"""

Modelling from scratch in Numpy gives a good understanding of underlying principles.
However, solving for and coding up backpropagation steps is quite tedious,
especially when you want to build deeper and more complex models.

Pytorch tensors offer us Numpy like arrays, along with two new components:
1. Dynamic Computational Graph -> Operations on tensors build the computational graph,
which allows instant backprop (gradient calculation) from any non-leaf scalar to any leaf tensor in the graph.
2. Support for GPU

The following piece of code builds a neural network capable of diverse supervised learning tasks.
It uses pytorch tensors as building blocks, through an object oriented methodology.

New Components:
-Batch Normalization (Finally! Made possible through Autograd)
-Save and Load model weights, which allows babysitting the training process
-Conditional Mean Paths - Predictions w.r.t a given feature, keeping all other features at their mean values.
This, under certain conditions, gives us the -direction and magnitude of the impact of a feature on final output.
-Use Best Results - Picks up the weights which deliver the best result on 1st Eval Metric

Older Components:
-Multiple Loss functions for classification, regression & multi-class.
-Alternative Activations - Relu, Swish, Sigmoid
-Minibatches, Careful Initialization,
-L1 and L2 regularization, Eval metrics - R2, Rmse, Accuracy, AUC
-GD, Momentum, RMSprop - ADAM Optimizer

"""

import numpy as np
import torch as t
import matplotlib.pyplot as plt
import matplotlib
import warnings

t.manual_seed(42)
plt.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)
warnings.filterwarnings("ignore")

class NeuralNetwork:
    """A multi-layered functional mapping between known inputs and known outputs,
    whose weights shift to minimize a suitable cost function. """

    def WeightInit(s, N_in, N_out):
        '''Scales weights in each layer.'''
        if s.init == 'small':
            return 0.01
        if s.init == 'He':
            return t.pow(t.tensor(2/N_in), 0.5)
        if s.init == 'Xavier':
            return t.pow(t.tensor(1/N_in), 0.5)
        if s.init == 'Benjamin':
            return t.pow(t.tensor(6/(N_in+N_out)), 0.5)
        if s.init == 'None':
            return 1

    def TrainValTestSplit(s, x, y, splits):
        '''Randomly splits inputs and labels into training, validation and test sets.'''
        m, perm = x.shape[0], t.randperm(x.shape[0])
        x, y = x[perm, :], y[perm, :]
        # Define split indices & split dataset
        a, b, c = splits[0]/sum(splits), splits[1]/sum(splits), splits[2]/sum(splits)
        s.x_train, s.x_val, s.x_test = x[0:round(a*m), :], x[round(a*m):round(a*m)+round(b*m), :], x[round(a*m)+round(b*m):, :]
        s.y_train, s.y_val, s.y_test = y[0:round(a*m), :], y[round(a*m):round(a*m)+round(b*m), :], y[round(a*m)+round(b*m):, :]

    def __init__(s, y, x,
                 hidden_layers = [10],
                 activations = ['relu', 'linear'],
                 objective = 'regression',
                 λ1 = 0,
                 λ2 = 0,
                 init = 'He',
                 seed = 42,
                 precision = 4,
                 miniBatchsize = 32,
                 eval_metrics = ['R2'],
                 splits = [80, 20, 0],
                 batch_norm = True,
                 feature_names = None,
                 use_best_result = None):

        # Network Architechture
        s.precision = precision
        t.manual_seed(seed)
        s.x, s.y = x, y
        s.TrainValTestSplit(x, y, splits)
        s.λ1, s.λ2 = λ1, λ2
        (s.m, s.n) = x.shape
        s.depth = len(hidden_layers) + 1
        s.c = y.shape[1]
        s.L = [s.n] + hidden_layers + [s.c]
        s.activations = activations
        s.miniBatchsize = miniBatchsize
        s.init = init
        s.objective = objective
        s.eval_metrics = eval_metrics
        s.train_loss, s.val_loss, s.test_loss = [], [], []
        s.loss = 1000000000
        s.batch_norm = batch_norm
        s.feature_names = feature_names
        s.use_best_result = use_best_result
        s.BestResult = None
        s.new = -1000000

        # Network Initialization
        s.A = {0:s.x_train}
        s.A.update({i:t.zeros(s.m, s.L[i]) for i in range(1, s.depth + 1)})
        s.W = {i: t.tensor(t.randn(s.L[i], s.L[i+1]) * s.WeightInit(s.L[i], s.L[i+1]), requires_grad = True)
               for i in range(s.depth)}
        s.dW = {i:t.zeros_like(s.W[i]) for i in range(s.depth)}
        s.mW = {i:t.zeros_like(s.W[i]) for i in range(s.depth)}
        s.VdW = {i:t.zeros_like(s.W[i]) for i in range(s.depth)}
        s.SdW = {i:t.zeros_like(s.W[i]) for i in range(s.depth)}
        s.b = {i:t.tensor(t.randn(1, s.L[i+1]) * s.WeightInit(s.L[i], s.L[i+1]), requires_grad = True)
               for i in range(s.depth)}
        s.db = {i:t.zeros_like(s.b[i]) for i in range(s.depth)}
        s.mb = {i:t.zeros_like(s.b[i]) for i in range(s.depth)}
        s.Vdb = {i:t.zeros_like(s.b[i]) for i in range(s.depth)}
        s.Sdb = {i:t.zeros_like(s.b[i]) for i in range(s.depth)}

        print('\nNEURAL NETWORK\n')
        print(f'\t Objective: {s.objective.title()}')
        print(f'\t Training, Validation, Test Examples - {s.x_train.shape[0], s.x_val.shape[0], s.x_test.shape[0]}')
        print(f'\t Raw Features - {s.n}')
        print(f'\t Final Output Classes - {s.c}')
        print(f'\t Architechture - {s.L}')
        print(f'\t Activations - {s.activations}')
        print(f'\t Depth - {s.depth}')
        print(f'\t Mini Batch Size - {s.miniBatchsize}')
        print(f'\t Init Type - {s.init}')
        print(f'\t Eval Metrics - {s.eval_metrics}')
        print(f'\t Random State - {seed}')
        print(f'\t L1 Regularization - {s.λ1}')
        print(f'\t L2 Regularization - {s.λ2}')
        print(f'\t Precision - {s.precision}')

    def Activation(s, z, i):
        '''Linear & Non-Linear Activation Functions'''
        if s.activations[i] == 'linear':
            return z
        if s.activations[i] == 'relu':
            return t.nn.functional.relu(z)
        if s.activations[i] == 'tanh':
            return t.nn.functional.tanh(z)
        if s.activations[i] == 'lrelu':
            return t.nn.functional.leaky_relu(z)
        if s.activations[i] == 'sigmoid':
            return t.nn.functional.sigmoid(z)
        if s.activations[i] == 'softmax':
            return t.nn.functional.softmax(z)

    def MiniBatchGenerator(s, m, size):
        '''Generates indices for where training examples are to be partitioned into mini-batches'''
        no_of_mbs = round(m/size)
        partitions = []
        idx_mbs = [i for i in range(no_of_mbs)]

        for i in idx_mbs:
            partitions.append([i*size, i*size + size])
        partitions[-1][1] = m + 1
        return partitions

    def ForwardPass(s, dropoutGrid):
        '''From Inputs calculates Output and Cost'''
        for i in range(0, s.depth):
            if s.batch_norm == True: # Batch Normalization
                μ = t.mean(s.A[i], axis = 0, keepdims = True)
                σ = t.std(s.A[i], axis = 0, keepdims = True)
                s.A[i] = (s.A[i] - μ + 0.000000001)/(σ+0.000000001)
            s.A[i+1] = s.Activation(t.mm(s.A[i], dropoutGrid[i]*s.W[i]) + s.b[i], i)

        # Regularization
        L1, L2 = 0, 0
        for i in s.W:
            L1 += t.mean(t.abs(s.W[i]))
            L2 += t.mean(t.pow(s.W[i], 2))
        paramCost = s.λ1*L1 + s.λ2*L2

        # Cost Function
        if s.objective == 'regression':
            s.loss = t.mean(t.pow(s.y_m - s.A[s.depth], 2)) + paramCost
        if s.objective == 'binary':
            s.loss = -t.mean(s.y_m*t.log(s.A[s.depth]+0.000000001) + (1-s.y_m)*t.log(1-s.A[s.depth]+0.000000001)) + paramCost
        if s.objective == 'multi-class':
            s.loss = -t.mean(s.y_m*t.log(s.A[s.depth]+0.000000001)) + paramCost

    def GradientDescent(s):
        ''' Obtain gradients for each weight, then update weights accordingly.'''
        s.loss.backward() # Backpropagation with Autograd!
        for i in range(s.depth - 1, -1, -1):
            s.dW[i] = s.W[i].grad
            s.db[i] = s.b[i].grad

        # Update ADAM parameters
        with t.no_grad():
            for i in range(s.depth):
                s.VdW[i] = s.mom * s.VdW[i] + (1-s.mom) * s.dW[i]
                s.SdW[i] = s.rmsprop * s.SdW[i] + (1-s.rmsprop) * t.pow(s.dW[i], 2)
                s.Vdb[i] = s.mom * s.Vdb[i] + (1-s.mom) * s.db[i]
                s.Sdb[i] = s.rmsprop * s.Sdb[i] + (1-s.rmsprop) * t.pow(s.db[i], 2)

        # Gradient Descent
        with t.no_grad():
            for i in range(s.depth):
                s.W[i] = t.tensor(s.W[i] - s.lrate * s.VdW[i]/t.pow(s.SdW[i]+s.ε, 0.5), requires_grad = True)
                s.b[i] = t.tensor(s.b[i] - s.lrate * s.Vdb[i]/t.pow(s.Sdb[i]+s.ε, 0.5), requires_grad = True)

    def Evaluation(s, y, yhat):
        '''Calculate Different Evaluation Metrics.'''
        result = []
        if 'SquaredError' in s.eval_metrics:
            squaredError = t.mean(t.pow(y - yhat, 2))
            result.append(['SquaredError:', round(squaredError.item(), s.precision)])
        if 'logloss' in s.eval_metrics:
            logloss = -t.mean(y*t.log(yhat+0.000000001) + (1-y)*t.log(1-yhat+0.000000001))
            result.append(['LogLoss:', round(logloss.item(), s.precision)])
        if 'softmaxloss' in s.eval_metrics:
            softmaxloss = -t.mean(y*t.log(yhat+0.000000001))
            result.append(['Softmax Loss:', round(softmaxloss.item(), s.precision)])
        if 'R2' in s.eval_metrics:
            RSS, TSS = t.mean(t.pow(y - yhat, 2)), t.mean(t.pow(y - t.mean(y), 2))
            R2 = 1 - RSS/TSS
            result.append(['R2:', round(R2.item(), s.precision)])
        if 'rmse' in s.eval_metrics:
            RSS = t.mean(t.pow(y - yhat, 2))
            rmse = t.pow(RSS,0.5)
            result.append(['rmse:', round(rmse.item(), s.precision)])
        if 'auc' in s.eval_metrics:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y.detach().numpy(), yhat.detach().numpy())
            result.append(['auc:', round(auc.item(), s.precision)])
        if 'binary-accuracy' in s.eval_metrics:
            pred = t.where(yhat > 0.5, t.ones(yhat.size()), t.zeros(yhat.size()))
            acc = t.mean(t.where(y == pred, t.ones(y.size()), t.zeros(y.size())))
            result.append(['bin-acc:', round(acc.item(), s.precision)])
        if 'auc' in s.eval_metrics:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y.detach().numpy(), yhat.detach().numpy())
            result.append(['auc:', round(auc.item(), s.precision)])
        if 'multi-accuracy' in s.eval_metrics:
            pred = t.argmax(yhat, axis = 1)
            actual = t.argmax(y, axis = 1)
            acc = t.mean(t.where(actual == pred, t.ones(pred.size()), t.zeros(pred.size())))
            result.append(['multi-accuracy:', round(acc.item(), s.precision)])
        return result

    def Train(s, lrate=0.1, iterations=100, mom=0.9, dropout_probs = None, rmsprop = 0.999, ε = 0.00000001, verbose = 10):
            '''Train the model through multiple iterations of forward and backward pass.'''

            batch = s.MiniBatchGenerator(s.x_train.shape[0], s.miniBatchsize)
            mb, epoch = 0, 0
            s.verbose = verbose
            s.iterations = iterations
            s.lrate = lrate
            s.mom = mom
            s.rmsprop = rmsprop
            s.ε = ε

            print(f'\t Iterations: {s.iterations}')
            print(f'\t Learning Rate - {s.lrate}')
            print(f'\t Momentum - {s.mom}')
            print(f'\t RMSProp - {s.rmsprop}')
            print(f'\t Epsilon - {s.ε} \n')

            for i in range(iterations):

                # Mini Batch & Dropout
                s.A[0] = s.x_train[batch[mb][0]:batch[mb][1], :]
                s.y_m = s.y_train[batch[mb][0]:batch[mb][1], :]
                if dropout_probs is None:
                    dropoutGrid = {i:t.where(t.Tensor(s.W[i].shape).uniform_(0, 1) <= 1,
                                             t.ones(s.W[i].shape), t.zeros(s.W[i].shape)) for i in s.W}
                else:
                    dropoutGrid = {i:t.where(t.Tensor(s.W[i].shape).uniform_(0, 1) < dropout_probs[i],
                                             t.ones(s.W[i].shape), t.zeros(s.W[i].shape)) for i in s.W}

                # Train
                s.ForwardPass(dropoutGrid)
                s.GradientDescent()

                # Evaluation
                y_train_pred, train_result = s.Score(s.y_train, s.x_train, scale=False)
                s.train_loss.append(train_result)

                if s.x_val.shape[0] != 0:
                    if s.batch_norm == True:
                        y_val_pred, val_result = s.Score(s.y_val, s.x_val)
                    else:
                        y_val_pred, val_result = s.Score(s.y_val, s.x_val, scale=False)
                    s.val_loss.append(val_result)

                if s.x_test.shape[0] != 0:
                    if s.batch_norm == True:
                        y_test_pred, test_result = s.Score(s.y_test, s.x_test)
                    else:
                        y_test_pred, test_result = s.Score(s.y_test, s.x_test, scale=False)
                    s.test_loss.append(test_result)

                if i >= 2*s.verbose:
                    s.UseBestResult(mode='save', i=i)

                # Print Evaluation
                if i % s.verbose == 0:
                    print(f'Iter {i}, Loss: {round(s.loss.item(), s.precision)}, Train Set: {train_result}')
                    if s.x_val.shape[0] != 0:
                        print(f'\t\t\t Val Set: {val_result}')
                    if s.x_test.shape[0] != 0:
                        print(f'\t\t\t Test Set: {test_result}')

                # Increase Mini-Batch & Epoch Count
                mb += 1
                if mb == len(batch):
                    mb, epoch = 0, epoch + 1

            # Use Best Result & Plot Learning Curves
            print('\n...Training Complete.')
            s.UseBestResult(mode='load', i=0)
            print('\nError Curves\n')
            s.ErrorCurves()

    def UseBestResult(s, mode, i):
        '''use_best_loss must be in format - [train/val/test, 0 for larger preffered].
        Will check if first eval metric is better or not, if so then will save it as best result'''

        if s.use_best_result is not None:
            old = s.new
            if mode=='save':
                if s.use_best_result[0] == 'train': # Select Metric for Comparision
                    s.var = s.train_loss
                elif s.use_best_result[0] == 'val':
                    s.var = s.val_loss
                else:
                    s.var = s.test_loss

                if ((s.use_best_result[1] == 0) & (s.new < s.var[-1][0][1])) or ((s.use_best_result[1] == 1) & (s.new < s.var[-1][0][1])):
                    s.new = s.var[-1][0][1]
                    s.iter = i
                    from copy import deepcopy as d # Compare, save Best Result
                    s.bestResult = [d(s.W), d(s.b), d(s.VdW), d(s.SdW), d(s.Vdb), d(s.Sdb)]

            elif ((mode=='load') & (s.use_best_result is not None)):
                with t.no_grad():
                    s.W, s.b, s.VdW, s.SdW, s.Vdb, s.Sdb = s.bestResult
                print(f'Using Best Result -> {s.use_best_result}, Iter:{s.iter}, {s.var[-1][0][0], s.new}')

    def SaveLoadWeights(s, mode):
        '''Save/Load Weights Temporarily'''
        if mode == 'save':
            from copy import deepcopy as d
            s.backup = [d(s.W), d(s.b), d(s.VdW), d(s.SdW), d(s.Vdb), d(s.Sdb)]
        elif mode == 'load':
            with t.no_grad():
                s.W, s.b, s.VdW, s.SdW, s.Vdb, s.Sdb = s.backup

    def ErrorCurves(s):
        import matplotlib.pyplot as plt
        for i in range(len(s.train_loss[0])):
            print('\t', s.train_loss[0][i][0])
            plt.plot([k[i][1] for k in s.train_loss], label = 'Train')
            plt.ylabel(f'{s.train_loss[0][i][0]}')
            plt.xlabel(f'Per {s.verbose} Iterations')
            plt.legend()
            if s.x_val.shape[0] != 0:
                plt.plot([k[i][1] for k in s.val_loss], label = 'Val')
                plt.ylabel(f'{s.val_loss[0][i][0]}')
                plt.xlabel(f'Per {s.verbose} Iterations')
                plt.legend()
            if s.x_test.shape[0] != 0:
                plt.plot([k[i][1] for k in s.test_loss], label = 'Test')
                plt.ylabel(f'{s.test_loss[0][i][0]}')
                plt.xlabel(f'Per {s.verbose} Iterations')
                plt.legend()
            plt.show()

    def ConditionalMeanPath(s, feature_indices=[1, 2]):
        print('''Conditional Mean Paths are defined as Yhat w.r.t. any given feature,
        when all other features are at their mean values.''')
        for k in feature_indices:
            x, y = s.x_train, s.y_train
            feature = t.linspace(x[:, k].min(), x[:, k].max(), 1000)
            avg_x = t.zeros((1000, x.shape[1]))

            for i in range(x.shape[1]):
                avg_x[:, i] = t.mean(x[:, i], axis = 0, keepdims = True)
                if i==k:
                    avg_x[:, k] = feature

            for i in range(s.depth):
                if (s.batch_norm == True): #& i > 0
                    μ = t.mean(avg_x, axis = 0, keepdims = True)
                    σ = t.std(avg_x, axis = 0, keepdims = True)
                    avg_x = (avg_x - μ + 0.00000001)/(σ + 0.00000001)
                avg_x = s.Activation(t.mm(avg_x, s.W[i]) + s.b[i], i)

            if s.feature_names is None:
                plt.scatter(feature.detach().numpy(), avg_x.detach().numpy(), label = k, s=10)
            else:
                plt.scatter(feature.detach().numpy(), avg_x.detach().numpy(), label = s.feature_names[k], s=10)

        plt.xlabel('Stdized Feature Value')
        plt.title('Conditional Mean Paths')
        plt.ylabel('Predicted Y')
        plt.legend()
        plt.show()

    def Score(s, y, x, scale = True):
        if scale == True:
            μ = t.mean(x, axis = 0, keepdims = True)
            σ = t.std(x, axis = 0, keepdims = True)
            x = (x - μ)/σ
        for i in range(s.depth):
            if s.batch_norm == True:
                μ = t.mean(x, axis = 0, keepdims = True)
                σ = t.std(x, axis = 0, keepdims = True)
                x = (x - μ)/(σ+0.000000001)
            x = s.Activation(t.mm(x, s.W[i]) + s.b[i], i)
        evaluation = s.Evaluation(y,x)
        return x.detach(), evaluation




"""

Benchmarking Algorithms:
- Sklearn Linear Regression
- Sklearn Logistic Regression
- Sklearn MLP Classifier
- XGBoost Classifier
- XGBoost Regressor

"""

def benchmark(x, y, feature_names, task='regression'):
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x.detach().numpy(), y.detach().numpy(), test_size = 0.2, random_state =42)

    if task == 'binary':
        print('\nBenchmarking to Popular Packages:\n')
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import log_loss, accuracy_score
        LR = LogisticRegression()
        LR.fit(x_train, y_train)
        print(f'Logloss for Sklearn Logistic Regression: {log_loss(y_test, LR.predict_proba(x_test)):0.4}')
        print(f'accuracy_score for Sklearn Logistic Regression: {accuracy_score(y_test, LR.predict(x_test)):0.4} \n')
        print('Coefficients:')
        print([(feature_names[i], np.round(LR.coef_[0][i], 2)) for i in range(len(feature_names))], '\n')

        from xgboost import XGBClassifier
        XGB = XGBClassifier(silent = True)
        XGB.fit(x_train, y_train)
        print(f'Logloss for XGboost: {log_loss(y_test, XGB.predict_proba(x_test)):0.4}')
        print(f'accuracy_score for XGboost: {accuracy_score(y_test, XGB.predict(x_test)):0.4} \n')

        from sklearn.neural_network import MLPClassifier
        MLP = MLPClassifier(alpha=0.1, max_iter=100)
        MLP.fit(x_train, y_train)
        print(f'Logloss for Sklearn MLP: {log_loss(y_test, MLP.predict(x_test)):0.4}')
        print(f'accuracy_score for Sklearn MLP: {accuracy_score(y_test, MLP.predict(x_test)):0.4} \n')

    if task == 'regression':
        print('\nBenchmarking to Popular Packages:\n')
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        LR = LinearRegression()
        LR.fit(x_train, y_train)
        print(f'r2_score Linear Regression: {r2_score(y_test, LR.predict(x_test)):0.4}')
        print(f'rmse Linear Regression: {np.sqrt(mean_squared_error(y_test, LR.predict(x_test))):0.4} \n')
        print('Coefficients:')
        print([(feature_names[i], np.round(LR.coef_[0][i], 2)) for i in range(len(feature_names))], '\n')

        from xgboost import XGBRegressor
        XGB = XGBRegressor(silent = True)
        XGB.fit(x_train, y_train)
        print(f'r2_score XGB: {r2_score(y_test, XGB.predict(x_test)):0.4}')
        print(f'rmse XGB: {np.sqrt(mean_squared_error(y_test, XGB.predict(x_test))):0.4} \n')

        from sklearn.neural_network import MLPRegressor
        MLP = MLPRegressor(alpha=0.1, max_iter=100)
        MLP.fit(x_train, y_train)
        print(f'r2_score MLP: {r2_score(y_test, MLP.predict(x_test)):0.4}')
        print(f'rmse XGB: {np.sqrt(mean_squared_error(y_test, MLP.predict(x_test))):0.4} \n')






"""

Benchmarking Datasets:
101 - Regression 1: Linear Regression on Concrete Dataset to predict Concrete Stength
102 - Regression 2: Non-Linear Regression on Concrete Dataset to predict Concrete Stength
103 - Classification 1: Logistic Regression for classifying Rich adults on US Census Data
104 - Classification 2: Non-Linear Classification for classifying Rich adults on US Census Data
105 - Multi-Classification 1: 10 Class, Synthetic Data
106 - Multi-Classification 2: SKlearn Digits Data

"""

print("""
Regression 1: Linear Regression on Concrete Dataset to predict Concrete Stength
Target - Concrete Strength
all features are measured in kg/m3, age in days (1-365) \n
""")

PATH = '/Users/pranjal/Google Drive/Projects/github/neural_network/tests/concrete.csv'
df = np.loadtxt(PATH, skiprows = 1, delimiter = ',')
y = t.tensor(df[:, -1:], dtype = t.float32)
x = t.tensor(df[:, :-1], dtype = t.float32)
x = (x - t.mean(x, axis = 0, keepdims = True) + 0.000000001)/(t.std(x, axis = 0, keepdims = True) + 0.000000001)

feature_names = ['cement', 'slag', 'flyash', 'water', 'superplastisizer', 'coarseaggregate', 'fineaggregate', 'age']
model1 = NeuralNetwork(y, x,
                      splits = [80, 20, 0],
                      hidden_layers = [],
                      activations = ['linear'],
                      objective = 'regression',
                      eval_metrics = ['R2', 'rmse'],
                      miniBatchsize = x.shape[0],
                      feature_names=feature_names,
                      batch_norm=False)
model1.Train(iterations = 100, lrate=0.1, mom=0.9, rmsprop=0.9999, verbose = 10)
benchmark(x, y, feature_names, task='regression')

print('''
Regression 2: Non-Linear Regression on Concrete Dataset to predict Concrete Stength
Target - Concrete Strength
All features are measured in kg/m3, age in days (1-365) \n
''')

PATH = '/Users/pranjal/Google Drive/Projects/github/neural_network/tests/concrete.csv'
df = np.loadtxt(PATH, skiprows = 1, delimiter = ',')
y = t.tensor(df[:, -1:], dtype = t.float32)
x = t.tensor(df[:, :-1], dtype = t.float32)

# Scale
x = (x-t.mean(x, axis = 0, keepdims = True) + 0.000000001)/(t.std(x, axis = 0, keepdims = True) + 0.000000001)

feature_names = ['cement', 'slag', 'flyash', 'water', 'superplastisizer', 'coarseaggregate', 'fineaggregate', 'age']
model2 = NeuralNetwork(y, x,
                      splits = [80, 20, 0],
                      hidden_layers = [50, 25, 10, 5],
                      activations = ['lrelu','lrelu', 'lrelu','lrelu', 'linear'],
                      objective = 'regression',
                      eval_metrics = ['R2', 'rmse'],
                      miniBatchsize = x.shape[0],
                      feature_names=feature_names,
                      λ2=3, use_best_result=['val', 0],
                      batch_norm=False)
model2.Train(iterations = 5000, lrate=0.01, mom=0.99, rmsprop=0.999, verbose = 50)
model2.SaveLoadWeights('save')
benchmark(x, y,feature_names, task = 'regression')

print('\nExploration of Direction & Magnitude of Influence of Features on Target\n')
print('Linear Regression - Conditional Mean Paths')
model1.ConditionalMeanPath([0, 1, 2, 3])
print('Non-Linear Regression - Conditional Mean Paths')
model2.ConditionalMeanPath([0, 1, 2, 3])

print('Linear Regression - Conditional Mean Paths')
model1.ConditionalMeanPath([4, 5, 6, 7])
print('Non-Linear Regression - Conditional Mean Paths')
model2.ConditionalMeanPath([4, 5, 6, 7])

print("""
Classification 1: Logistic Regression for classifying Rich adults on US Census Data
Target - Income above 50$ (negative) or not (positive)
Features - over 100, including dummies. \n
""")

import pandas as pd
from pmlb import fetch_data
df = fetch_data('adult')

# Dummify
one_hot_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
for i in one_hot_encode:
    df = pd.concat([df, pd.get_dummies(df[i], prefix=i, drop_first=True)], axis = 1)
df.drop(one_hot_encode, axis = 1, inplace = True)

x = t.tensor(np.array(df.drop('target', axis =1)), dtype = t.float32)
y = t.tensor(np.array(df[['target']]), dtype = t.float32)
x = (x-t.mean(x, axis = 0, keepdims = True) + 0.000000001)/(t.std(x, axis = 0, keepdims = True)+0.000000001)

# Target - Rich (1) or not (0)
feature_names=list(df.drop('target', axis =1).columns)
model3 = NeuralNetwork(y, x, splits = [80, 20, 0],
                      hidden_layers = [],
                      activations = ['sigmoid'],
                      objective = 'binary',
                      eval_metrics = ['logloss', 'binary-accuracy'],
                      miniBatchsize = x.shape[0],
                      feature_names = feature_names,
                      batch_norm=False)

model3.Train(iterations = 100, lrate=0.01, verbose = 1)
benchmark(x, y,feature_names, task='binary')

print('\nExploration of Direction & Magnitude of Influence of Features on Target\n')
model3.ConditionalMeanPath([0, 1, 2])
model3.ConditionalMeanPath([3, 4, 5])

print("""
Classification 2: Non Linear Model for classifying Rich adults on US Census Data
Target - Income above 50$ (negative) or not (positive)
Features - over 100, including dummies. \n
""")

import pandas as pd
from pmlb import fetch_data
df = fetch_data('adult')

# Dummify
one_hot_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
for i in one_hot_encode:
    df = pd.concat([df, pd.get_dummies(df[i], prefix=i, drop_first=True)], axis = 1)
df.drop(one_hot_encode, axis = 1, inplace = True)

x = t.tensor(np.array(df.drop('target', axis =1)), dtype = t.float32)
y = t.tensor(np.array(df[['target']]), dtype = t.float32)

feature_names=list(df.columns)
model4 = NeuralNetwork(y, x,
                      splits = [80, 20, 0],
                      hidden_layers = [75, 50, 25, 15, 10, 10],
                      activations = ['lrelu','lrelu','lrelu','lrelu','lrelu','lrelu','sigmoid'],
                      objective = 'binary',
                      eval_metrics = ['binary-accuracy'],
                      miniBatchsize= x.shape[0],
                      feature_names = feature_names,
                      batch_norm=True,
                      λ1=1,
                      use_best_result=['val', 0])
model4.Train(iterations = 200, lrate=0.01, mom=0.9, rmsprop=0.999, verbose = 1)

print("""Multi-Classification 1: 10 Class, Synthetic Data""")
from sklearn.datasets import make_classification
x, y = make_classification(n_samples=10000, n_features=100, n_informative=50, n_redundant=10,
                         n_repeated=10, n_classes=5, random_state=42)
y = np.c_[np.where(y == 0, 1, 0),
          np.where(y == 1, 1, 0),
          np.where(y == 2, 1, 0),
          np.where(y == 4, 1, 0),
          np.where(y == 5, 1, 0)]
x = t.tensor(x, dtype=t.float32)
y = t.tensor(y, dtype=t.float32)
model5 = NeuralNetwork(y, x,
                      splits = [80, 20, 0],
                      hidden_layers = [120, 30],
                      activations = ['lrelu','lrelu','softmax'],
                      objective = 'multi-class',
                      eval_metrics = [ 'multi-accuracy'],
                      miniBatchsize=x.shape[0],
                      λ2=0.1,
                      batch_norm = True,
                      use_best_result=['val', 0])
model5.Train(iterations = 200, lrate=0.01, mom=0.9, rmsprop=0.999, verbose = 1)

print('Compare with XGB')
from sklearn.metrics import accuracy_score as acc
x, y = make_classification(n_samples=10000, n_features=100, n_informative=50, n_redundant=10,
                         n_repeated=10, n_classes=5, random_state=42)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =42)
from xgboost import XGBClassifier
XGB = XGBClassifier(silent = True)
XGB.fit(x_train, y_train)
print(f'accuracy_score for XGboost: {acc(y_test, XGB.predict(x_test)):0.4} \n')

print("""Multi-Classification 2: Sklearn MNIST Digits""")
from sklearn.datasets import load_digits
x, y = load_digits(return_X_y=True)
y = np.c_[np.where(y == 0, 1, 0),
          np.where(y == 1, 1, 0),
          np.where(y == 2, 1, 0),
          np.where(y == 3, 1, 0),
          np.where(y == 4, 1, 0),
          np.where(y == 5, 1, 0),
          np.where(y == 6, 1, 0),
          np.where(y == 7, 1, 0),
          np.where(y == 8, 1, 0),
          np.where(y == 9, 1, 0)]
x = t.tensor(x, dtype=t.float32)
y = t.tensor(y, dtype=t.float32)
model6 = NeuralNetwork(y, x,
                      splits = [80, 20, 0],
                      hidden_layers = [500, 300, 100, 50],
                      activations = ['lrelu','lrelu','lrelu','lrelu','softmax'],
                      objective = 'multi-class',
                      eval_metrics = [ 'multi-accuracy'],
                      miniBatchsize=x.shape[0],
                      batch_norm=False , λ2=10,
                      use_best_result=['val', 0])
model6.Train(iterations = 500, lrate=0.0001, mom=0.9, rmsprop=0.999, verbose = 1, dropout_probs = [0.9, 0.9, 0.9, 0.9, 1])

print('Compare with XGB')
from sklearn.datasets import load_digits
x, y = load_digits(return_X_y=True)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
from xgboost import XGBClassifier
XGB = XGBClassifier(silent = True)
XGB.fit(x_train, y_train)
print(f'accuracy_score for XGboost: {acc(y_test, XGB.predict(x_test)):0.4} \n')
