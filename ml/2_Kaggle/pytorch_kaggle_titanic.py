#############################################################
## Predicting Survival of Passengers on the Titanic
## Public Leader board score -> 82.7%, Rank at that time 371/16k (Top 3%)
## Data: https://www.kaggle.com/c/titanic/data
#############################################################

# Load the data
test_path = '/Users/pranjal/Google Drive/Projects/github/kaggle_pytorch/titanic/test.csv'
train_path = '/Users/pranjal/Google Drive/Projects/github/kaggle_pytorch/titanic/train.csv'
init_sub_path = '/Users/pranjal/Google Drive/Projects/github/kaggle_pytorch/titanic/gender_submission.csv'
final_sub_path = '/Users/pranjal/Google Drive/Projects/github/kaggle_pytorch/titanic/submission.csv'

import pandas as pd, numpy as np
sub = pd.read_csv(init_sub_path)
test = pd.read_csv(test_path)
train = pd.read_csv(train_path)

# Indicator
train['data'] = 'train'
test['data'] = 'test'

# Join Train & Test
df = train.append(test)

######################################
## Basic PreProcessing, Basic Features
######################################

# Surname, Title, Family Size
df['Surname'] = df.Name.str.split(', ').str[0]
df['Title'] = df.Name.str.split(', ').str[1].str.split('. ').str[0]
df.Title.str.replace('Mme', 'Miss')
df.Title.str.replace('Ms', 'Miss')
df.Title.str.replace('Mlle', 'Miss')
df.loc[~df.Title.isin(['Master', 'Mr', 'Miss', 'Mrs']), 'Title'] = 'Rare'
df['Family'] = df['SibSp'] + df['Parch'] + 1

# Gender
df['Gender'] = 0
df.loc[(df.Sex == 'female'), 'Gender'] = 1

# Fare
df.loc[(df['Fare'] == 0) & (df['Pclass'] == 1), 'Fare'] = df[df['Pclass'] == 1].Fare.mode()[0]
df.loc[(df['Fare'] == 0) & (df['Pclass'] == 2), 'Fare'] = df[df['Pclass'] == 2].Fare.mode()[0]
df.loc[(df['Fare'] == 0) & (df['Pclass'] == 3), 'Fare'] = df[df['Pclass'] == 3].Fare.mode()[0]
df['Fare_Per_Person'] = 1 + df['Fare'] / df['Family']

###############################
## CatBoost Model for accurate Age Prediction
###############################

# Feature Selection
features = ['Title', 'Family', 'Gender', 'Pclass', 'Fare']
cat_features = [0, 3]

# Training & Scoring datasets
model_train, model_test = df[df.Age.notna()], df[~df.Age.notna()]
x_train, y_train = model_train[features], model_train[['Age']].astype('float')
x_test = model_test[features]

# Null handling
x_train = x_train.fillna(-999)
x_test = x_test.fillna(-999)

# Train
from catboost import CatBoostRegressor, Pool
train_pool = Pool(x_train, y_train, cat_features)
clf = CatBoostRegressor(random_state = 42, eval_metric = 'R2', verbose = 100)
clf.fit(train_pool)

# Score missing Age values
ypred = clf.predict(x_test)
df.loc[df.Age.isnull(), 'Age'] = ypred

#########################################
## Woman & Child Groups: Advanced Features
#########################################

# Child
print('WCG Group Features')
df['Child'] = 0
df.loc[(df.Title.str.contains('Master')) | (df.Age < 16), 'Child'] = 1

# WCG Group Identifier
df["FamilyGroup"] = df["Pclass"].astype(str) + "~" \
                   + df["Ticket"].str[:-1] + "~" \
                     + df["Embarked"] + "~" \
                        + df["Fare"].astype(str)

# WCG Counts in Train and Test 
WCG_all = df.loc[(df["Gender"] == 1) | (df["Child"] == 1)].groupby("FamilyGroup").count()[["PassengerId"]]
WCG_all.columns = ['WCG_TOT_CNT']
print('WCG Total', WCG_all.shape)
WCG_all.head()

# WCG Counts, Survived in Train  
train = df[df.data=='train']
WCG_cnt = train.loc[(train["Gender"] == 1) | (train["Child"] == 1)].groupby("FamilyGroup").count()[["PassengerId"]]
WCG_rate = train.loc[(train["Gender"] == 1) | (train["Child"] == 1)].groupby("FamilyGroup").sum()[["Survived"]]
WCG_train = pd.merge(WCG_cnt, WCG_rate, how="inner", on="FamilyGroup")
WCG_train.columns = ['WCG_TRAIN_CNT', 'WCG_TRAIN_SURV_CNT']
print('WCG Train', WCG_train.shape)

# Test 
test = df[df.data=='test']
WCG_test = test.loc[(test["Gender"] == 1) | (test["Child"] == 1)].groupby("FamilyGroup").count()[["PassengerId"]]
WCG_test.head()
WCG_test.columns = ['WCG_TEST_CNT']
print('WCG Test', WCG_test.shape)

# WCG -> Counts in Total, Train, Test and Train Survival Rates
WCG = pd.merge(pd.merge(WCG_all, WCG_train, how = 'left', on = 'FamilyGroup'), WCG_test, how = 'left', on = 'FamilyGroup')
WCG["WCG_SRate"] = WCG["WCG_TRAIN_SURV_CNT"] / WCG["WCG_TRAIN_CNT"]
WCG.sort_values(by = "WCG_SRate", inplace = True)
WCG = WCG.reset_index()
print(WCG.head())
print(WCG.tail())

import matplotlib.pyplot as plt
print('WCG Survival Rates Histogram')
print('Either all live, or all die')
print(WCG['WCG_SRate'].hist(bins = 10))

# Removing Solo Travellers
WCG = WCG[WCG["WCG_TOT_CNT"] > 1]
WCG_list = list(WCG.FamilyGroup)

# Join WCG Features
df = pd.merge(df, WCG, how="left", on="FamilyGroup")

# WCG vs Non-WCG Data
df['approach'] = 'model'
df.loc[(df.Gender == 0) & (df.Child == 1) & (df["WCG_SRate"] >= 0.5), 'approach'] = 'rule'
df.loc[(df.Gender == 1) & (df["WCG_SRate"] == 0), 'approach'] = 'rule'

WCG_df, xWCG_df = df[df.approach=='rule'], df[df.approach=='model']
print(WCG_df.shape, xWCG_df.shape)

###############################
## WCG Test Data: Simple Rules
###############################

# Rule based prediction
# Male Children with WCG Families which have high survival rate will survive
# Women with WCG Families which have high survival rate will not survive
WCG_df['pred'] = 0
WCG_df.loc[(WCG_df.Gender == 0) & (WCG_df["WCG_SRate"] == 1), 'pred'] = 1
WCG_df.loc[(WCG_df.Gender == 1) & (WCG_df["WCG_SRate"] == 0), 'pred'] = 0


##############################################
## Non-WCG Test Data: PyTorch Deep Neural Net
##############################################

# PyTorch Deep Neural Net
features = ['Title', 'Family', 'Gender', 'Pclass', 'Fare', 'Age'] 
cat_features = ['Title', 'Pclass']
model_train, model_test = xWCG_df[xWCG_df.Survived.notna()], xWCG_df[~xWCG_df.Survived.notna()]
x_train, y_train = model_train[features], model_train.Survived
x_test, y_test = model_test[features], model_test.Survived

# Scale & Encode
def PreProcess(df, cat_features):
    # Features Scaling
    ε = 0.0000000001
    for i in [i for i in features if i not in cat_features]:
        (df[i] - df[i].mean(axis = 0) + ε)/(df[i].std(axis = 0) + ε)
    
    # One Hot Encode
    for i in cat_features:
        df = pd.concat([df, pd.get_dummies(df[i], prefix = f'{i}_')], axis = 1)
        df.drop(i, axis = 1, inplace = True)
        
    return df

x_train = PreProcess(x_train, cat_features)
x_test = PreProcess(x_test, cat_features)

# Train-Val Separation
import torch as t
t.manual_seed(3)
y = t.tensor(y_train.values, dtype=t.float)
x = t.tensor(x_train.values, dtype=t.float)

train_size = 0.8
training_rows = t.LongTensor(round(x.shape[0]*train_size)).random_(0, x.shape[0])
validation_rows = [i for i in list(range(x.shape[0])) if i not in training_rows]
x_train, y_train = x[training_rows], y[training_rows]
x_val, y_val = x[validation_rows], y[validation_rows]

# Network Architechture
n_input, n_output, hidden = x_train.shape[1], 1, x_train.shape[1] + 10
neuralNet = t.nn.Sequential(
    t.nn.Linear(n_input, hidden),
    t.nn.ReLU(),
    t.nn.Linear(hidden, hidden),
    t.nn.ReLU(),
    t.nn.Linear(hidden, hidden),
    t.nn.ReLU(),
    t.nn.Linear(hidden, hidden),
    t.nn.ReLU(),
    t.nn.Linear(hidden, hidden),
    t.nn.ReLU(),
    t.nn.Linear(hidden, round(hidden/2)),
    t.nn.ReLU(),
    t.nn.Linear(round(hidden/2), n_output),
    t.nn.Sigmoid()
)

# Optimizer, Weights, Loss Function
optimizer = t.optim.Adam(neuralNet.parameters(), lr=0.001, weight_decay = 0.02)
weight = t.tensor(y_train)
weight = t.where(weight >= 0.5, t.tensor(1.0), t.tensor(1.0))
loss_fn = t.nn.BCELoss(weight=weight, reduction='mean')

# Evaluation Metric
from sklearn.metrics import accuracy_score
def eval_metric(a, b):
    pred = t.round(a)
    return round(accuracy_score(pred.detach().numpy(), b.detach().numpy()), 4)

# Train the Network
for i in range(5000):
    loss = loss_fn(neuralNet(x_train), y_train)
    if i % 1000 == 0:
        print(i, round(loss.item(), 2), eval_metric(neuralNet(x_train), y_train), eval_metric(neuralNet(x_val), y_val))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Score and Impute
ypred = neuralNet(t.tensor(x_test.values, dtype = t.float))
model_test['pred'] = np.where(ypred.detach().numpy() > 0.2, 1, 0)

# CatBoost Classifier 
#model_train, model_test = xWCG_df[xWCG_df.Survived.notna()], xWCG_df[~xWCG_df.Survived.notna()]
#features = ['Title', 'Family', 'Gender', 'Pclass', 'Fare', 'Age'] 
#cat_features = [0, 3]
#x_train, y_train = model_train[features], model_train.Survived
#x_test = model_test[features]
#x_train = x_train.fillna(-99)
#x_test = x_test.fillna(-99)
#from catboost import CatBoostClassifier, Pool
#train_pool = Pool(x_train, y_train, cat_features)
#clf = CatBoostClassifier(verbose = 100, eval_metric = 'Accuracy')
#clf.fit(train_pool)
#ypred = clf.predict(x_test)
#model_test['pred'] = ypred.astype('int')

##############################################
## Combine WCG, non-WCG scores and Export
##############################################

# Combine Scores & Export
scores = pd.concat([rule[['PassengerId', 'pred']], model_test[['PassengerId', 'pred']]], axis = 0)
sub_new = pd.merge(scores, sub[['PassengerId']], on = 'PassengerId')
sub_new.columns = ['PassengerId', 'Survived']
sub_new = sub_new.sort_values(by = 'PassengerId')
print(sub_new.Survived.sum())
sub_new.to_csv(final_sub_path, index = False)
