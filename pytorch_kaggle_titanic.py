test_path = '/Users/pranjal/Google Drive/Projects/github/kaggle_pytorch/titanic/test.csv'
train_path = '/Users/pranjal/Google Drive/Projects/github/kaggle_pytorch/titanic/train.csv'
sub_path = '/Users/pranjal/Google Drive/Projects/github/kaggle_pytorch/titanic/gender_submission.csv'
new_sub_path = '/Users/pranjal/Google Drive/Projects/github/kaggle_pytorch/titanic/submission.csv'

import pandas as pd, numpy as np
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

sub = pd.read_csv(sub_path)
test = pd.read_csv(test_path)
train = pd.read_csv(train_path)

# Train Test Indicator
train['data'] = 'train'
test['data'] = 'test'

df = train.append(test)
df = df.replace('', np.nan)
df.head()


'''
Basic Feature Engineering
'''

# Surname
df['Surname'] = df.Name.str.split(', ').str[0]

# Title
df['Title'] = df.Name.str.split(', ').str[1].str.split('. ').str[0]

# Gender
df['Gender'] = 0
df.loc[(df.Sex == 'female'), 'Gender'] = 1

# Family Size
df['family_size'] = 1 + df['SibSp'] + df['Parch']

# Class based features
df['Pclass1_W'] = 0
df.loc[(df.Pclass == 1) & (df.Gender == 1), 'Pclass1_W'] = 1
df['Pclass2_W'] = 0
df.loc[(df.Pclass == 2) & (df.Gender == 1), 'Pclass2_W'] = 1
df['Pclass3_W'] = 0
df.loc[(df.Pclass == 3) & (df.Gender == 1), 'Pclass3_W'] = 1
df['Pclass1_M'] = 0
df.loc[(df.Pclass == 1) & (df.Gender == 1), 'Pclass1_M'] = 1
df['Pclass2_M'] = 0
df.loc[(df.Pclass == 2) & (df.Gender == 1), 'Pclass2_M'] = 1
df['Pclass3_M'] = 0
df.loc[(df.Pclass == 3) & (df.Gender == 1), 'Pclass3_M'] = 1

# Unmarried Woman
df['Unmarried_Woman'] = 0
df.loc[(df.Name.str.contains('Miss.')) | (df.Name.str.contains('Mll.')) | (df.Name.str.contains('Ms.')) , 'Unmarried_Woman'] = 1

# Intermediate Features
def clean_categorical(data, dirty_categorical, values_desired):
    for i in dirty_categorical:
        appender = []
        for j in np.array(data[i]).astype(str):           
                if j[0] in values_desired: appender.append(1)
                else: appender.append(0)
        var1 = str(i) + '_notnull'
        data[var1] = appender
    
        appender=[]   
        for j in np.array(data[i]).astype(str):
            if j == 'nan':
                appender.append('None')
            elif [j][0][0] in values_desired:
                appender.append([j][0][0])
            else: appender.append('Other')
        var2 = str(i) + '_clean'
        data[var2] = appender

    
 # Cabin
clean_categorical(df, ['Cabin'], ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'T'])
df = pd.concat([df, pd.get_dummies(df.Cabin_clean, columns=['type'])], axis = 1)
cabin_array = np.array(df['Cabin']).astype(str)
cabin_count = []
for i in cabin_array:
    if (i == 'None'):
        cabin_count.append(0)
    elif (len(i) in (2,3,4)): 
        cabin_count.append(1)
    elif (len(i) in (5,6,7,8,9)):
        cabin_count.append(2)
    else: cabin_count.append(3)
df['no_cabins'] = cabin_count

# Fare - Some values were 0, others null.
#df['Fare_update_dummy'] = 0
#df.loc[(df['Fare'] == 0), 'Fare_update_dummy'] = 1
#df.loc[(df['Fare'] == 0) & (df['Pclass'] == 1), 'Fare'] = 87.508992
#df.loc[(df['Fare'] == 0) & (df['Pclass'] == 2), 'Fare'] = 21.179196
#df.loc[(df['Fare'] == 0) & (df['Pclass'] == 3), 'Fare'] = 13.302889
df['fare_persons'] = 1 + df['Fare'] / df['family_size']

# Null, Blank String Handling
df.Age = df.replace(df.Age.mode()[0])
df.Age = df.replace(df.Fare.mode()[0])
df.Age = df.replace(df.Embarked.mode()[0])

# Main Title: One of 5 Categories
import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

df['MainTitle'] = df['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
df['MainTitle'] = df['MainTitle'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
df['MainTitle'] = df['MainTitle'].replace('Mlle', 'Miss')
df['MainTitle'] = df['MainTitle'].replace('Ms', 'Miss')
df['MainTitle'] = df['MainTitle'].replace('Mme', 'Mrs')
df['MainTitle'] = df['MainTitle'].fillna('None')
df = pd.concat([df, pd.get_dummies(df.MainTitle, columns=['type'])], axis = 1)

# Character Types
df['woman_child'] = 0
df.loc[(df['Sex'] == 'female') | (df['Age'] < 15), 'woman_child'] = 1

df['rich_old_man'] = 0
df.loc[(df['Sex'] == 'male') & (df['Age'] > 45) & (df['Pclass'] == 1), 'rich_old_man'] = 1

df['poor_young_man'] = 0
df.loc[(df['Sex'] == 'male') & (df['Age'] > 20) & (df['Age'] < 30) & (df['Pclass'] == 3), 'poor_young_man'] = 1

df['alone_man'] = 0
df.loc[(df['Sex'] == 'male') & (df['family_size'] == 1), 'alone_man'] = 1

df['alone_woman'] = 0
df.loc[(df['Sex'] == 'female') & (df['family_size'] == 1), 'alone_woman'] = 1

list2 = ['E', 'D', 'B']
df['man_close_to_escape'] = 0
df.loc[(df['Sex'] == 'M') & (df['Cabin_clean'].isin(list2)), 'man_close_to_escape'] = 1


'''
Complex Features - Age, WCG Group
'''

df_new = df.copy()
features = ['Master', 'family_size', 'Miss', 'Pclass']

# Train - Test/Holdout Split
model_train, model_test = df[df.Age.notna()], df[~df.Age.notna()]
x_train, y_train = model_train[features], model_train[['Age']].astype('float')
x_test, y_test = model_test[features], model_test[['Age']].astype('float')

x_train= x_train.fillna(-999)
x_test = x_test.fillna(-999)

cat_features = list(np.where(x_train.dtypes == 'object')[0])
x_train, y_train = model_train[features], model_train.Age
x_test, y_test = model_test[features], model_test.Age

from sklearn.model_selection import train_test_split
x_train2, x_val, y_train2, y_val = train_test_split(x_train, y_train, test_size = 0.8)
cat_features = []
from catboost import CatBoostRegressor, Pool
train_pool = Pool(x_train, y_train, cat_features)
val_pool = Pool(x_val, y_val, cat_features)
test_pool = Pool(x_test, y_test, cat_features)
clf = CatBoostRegressor(iterations = 200, max_depth = 3, random_state = 42, eval_metric = 'R2', verbose = 500)
clf.fit(train_pool, eval_set = val_pool)
ypred = clf.predict(test_pool)
df.loc[df.Age.isnull(), 'Age'] = ypred


# Child
df['Child'] = 0
df.loc[(df.Title.str.contains('Master')) | (df.Age < 18), 'Child'] = 1

# WCG Group
df["FamilyGroup"] = df["Pclass"].astype(str) + " - " + df["Ticket"].str[:-1] + " - " + df["Embarked"] + " - " + df["Fare"].astype(str)

# Train + Test 
WCG_all = df.loc[(df["Gender"] == 1) | (df["Child"] == 1)].groupby("FamilyGroup").count()[["PassengerId"]]
WCG_all.columns = ['Total WCG Count']
print(WCG_all.shape)

# Train 
train = df[df.data=='train']
WCG_cnt = train.loc[(train["Gender"] == 1) | (train["Child"] == 1)].groupby("FamilyGroup").count()[["PassengerId"]]
WCG_rate = train.loc[(train["Gender"] == 1) | (train["Child"] == 1)].groupby("FamilyGroup").sum()[["Survived"]]
WCG_train = pd.merge(WCG_cnt, WCG_rate, how="inner", on="FamilyGroup")
WCG_train.columns = ['Train WCG Count', 'Train WCG Survived Count']
print(WCG_train.shape)

# Test 
test = df[df.data=='test']
WCG_test = test.loc[(test["Gender"] == 1) | (test["Child"] == 1)].groupby("FamilyGroup").count()[["PassengerId"]]
WCG_test.head()
WCG_test.columns = ['Test WCG Count']
print(WCG_test.shape)

# Train - Test Combine
WCG = pd.merge(pd.merge(WCG_all, WCG_train, how = 'left', on = 'FamilyGroup'), WCG_test, how = 'left', on = 'FamilyGroup')
WCG["Train Survival Rate"] = WCG["Train WCG Survived Count"] / WCG["Train WCG Count"]
WCG = WCG.reset_index()
print(WCG.shape)

WCG['Train Survival Rate'].hist(bins = 10)

# Removing Solo Travellers
WCG_familyGroups = WCG[WCG["Total WCG Count"] > 1]
WCG_familyGroups_list = list(WCG_familyGroups.FamilyGroup)
WCG_familyGroups.head()

'''
Model Building

'''

# Inital Prediction All Die, this gives 67% accuracy
sub['Survived'] = 0
print(sub.Survived.sum(), sub.Survived.count())

# All Women Survive, this gives 79% Accuracy. 
sub.loc[test["Gender"] == 1, 'Survived'] = 1
print(sub.Survived.sum(), sub.Survived.count(), sub.Survived.mean())

# Break Dataset into what will be dealt through 'WCG Rules' and what will not. 
df2 = pd.merge(df, WCG_familyGroups, how="left", on="FamilyGroup")
df2['approach'] = 'model'
df2.loc[(df2.Gender == 0) & (df2.Child == 1) & (df2["Train Survival Rate"] >= 0.5), 'approach'] = 'rule'
df2.loc[(df2.Gender == 1) & (df2["Train Survival Rate"] == 0), 'approach'] = 'rule'
rule, model = df2[df2.approach=='rule'], df2[df2.approach=='model']
print(rule.shape, model.shape)

# Rule based prediction
# Male Children with WCG Families which have high survival rate will survive
# Women with WCG Families which have high survival rate will not survive
rule['pred'] = 0
rule.loc[(rule.Gender == 0) & (rule["Train Survival Rate"] == 1), 'pred'] = 1
rule.loc[(rule.Gender == 1) & (rule["Train Survival Rate"] == 0), 'pred'] = 0
rule.head()

def featureScale(df):
    e = 0.0000000001
    return (df - df.mean(axis = 0) +  e)/ (df.std(axis = 0)+ e)

features = ['Age', 'Fare', 'Child','family_size',#'fare_persons',
            'Pclass1_W','Pclass2_W','Pclass3_W','Pclass1_M','Pclass2_M','Pclass3_M',
            'Unmarried_Woman','Total WCG Count','Train WCG Survived Count','Train Survival Rate',
            'Cabin_notnull', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'None', 'T', 'no_cabins',
             'Master', 'Miss', 'Mr', 'Mrs', 'Rare', 'rich_old_man', 'poor_young_man',
            'alone_man', 'alone_woman', 'man_close_to_escape']
features = ['Train Survival Rate', 'Gender', 'Age']
            
            
model_train, model_test = rule[rule.Survived.notna()], rule[~rule.Survived.notna()]
x_train, y_train = model_train[features], model_train.Survived
x_test, y_test = model_test[features], model_test.Survived
x_train = x_train.fillna(-10)
x_test = x_test.fillna(-10)
#x_train = featureScale(x_train)
#x_test = featureScale(x_test)


# Validation
import torch as t
t.manual_seed(42)
y = t.tensor(y_train.values, dtype=t.float)
x = t.tensor(x_train.values, dtype=t.float)

train_size = 0.7
training_rows = t.LongTensor(round(x.shape[0]*train_size)).random_(0, x.shape[0])
validation_rows = [i for i in list(range(x.shape[0])) if i not in training_rows]
x_train, y_train = x[training_rows], y[training_rows]
x_val, y_val = x[validation_rows], y[validation_rows]

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
optimizer = t.optim.Adam(neuralNet.parameters(), lr=0.01, weight_decay = 0.02)
weight = t.tensor(y_train)
weight = t.where(weight >= 0.5, t.tensor(1.0), t.tensor(1.5))
loss_fn = t.nn.BCELoss(weight=weight, reduction='mean')

from sklearn.metrics import accuracy_score
def eval_metric(a, b):
    pred = t.round(a)
    return round(accuracy_score(pred.detach().numpy(), b.detach().numpy()), 4)

# Train
for i in range(5000):
    loss = loss_fn(neuralNet(x_train), y_train)
    if i % 100 == 0:
        print(i, round(loss.item(), 2), eval_metric(neuralNet(x_train), y_train), eval_metric(neuralNet(x_val), y_val))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

ypred = neuralNet(t.tensor(x_test.values, dtype = t.float))
rule.loc[rule.Survived.isnull(), 'pred'] = np.where(ypred.detach().numpy() > 0.5, 1, 0)


# Model Based Predictions
# Neural Network
def featureScale(df):
    e = 0.0000000001
    return (df - df.mean(axis = 0) +  e)/ (df.std(axis = 0)+ e)

features = ['Age', 'Fare', 'Child','family_size',#'fare_persons',
            'Pclass1_W','Pclass2_W','Pclass3_W','Pclass1_M','Pclass2_M','Pclass3_M',
            'Unmarried_Woman','Total WCG Count','Train WCG Survived Count','Train Survival Rate',
            'Cabin_notnull', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'None', 'T', 'no_cabins',
             'Master', 'Miss', 'Mr', 'Mrs', 'Rare', 'rich_old_man', 'poor_young_man',
            'alone_man', 'alone_woman', 'man_close_to_escape']
features = ['Train Survival Rate', 'Gender', 'Age']
            
            
model_train, model_test = rule[rule.Survived.notna()], rule[~rule.Survived.notna()]
x_train, y_train = model_train[features], model_train.Survived
x_test, y_test = model_test[features], model_test.Survived
x_train = x_train.fillna(-10)
x_test = x_test.fillna(-10)
#x_train = featureScale(x_train)
#x_test = featureScale(x_test)


# Validation
import torch as t
t.manual_seed(42)
y = t.tensor(y_train.values, dtype=t.float)
x = t.tensor(x_train.values, dtype=t.float)

train_size = 0.7
training_rows = t.LongTensor(round(x.shape[0]*train_size)).random_(0, x.shape[0])
validation_rows = [i for i in list(range(x.shape[0])) if i not in training_rows]
x_train, y_train = x[training_rows], y[training_rows]
x_val, y_val = x[validation_rows], y[validation_rows]

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
optimizer = t.optim.Adam(neuralNet.parameters(), lr=0.01, weight_decay = 0.02)
weight = t.tensor(y_train)
weight = t.where(weight >= 0.5, t.tensor(1.0), t.tensor(1.5))
loss_fn = t.nn.BCELoss(weight=weight, reduction='mean')

from sklearn.metrics import accuracy_score
def eval_metric(a, b):
    pred = t.round(a)
    return round(accuracy_score(pred.detach().numpy(), b.detach().numpy()), 4)

# Train
for i in range(5000):
    loss = loss_fn(neuralNet(x_train), y_train)
    if i % 100 == 0:
        print(i, round(loss.item(), 2), eval_metric(neuralNet(x_train), y_train), eval_metric(neuralNet(x_val), y_val))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

ypred = neuralNet(t.tensor(x_test.values, dtype = t.float))
   

# CatBoost based
#model_train, model_test = model[model.Survived.notna()], model[~model.Survived.notna()]
#features = ['Pclass', 'Gender', 'Age', 'Child', 'family_size', 'Fare']
#x_train, y_train = model_train[features], model_train.Survived
#x_test, y_test = model_test[features], model_test.Survived

#x_train= x_train.fillna(-99)
#x_test = x_test.fillna(-99)
#cat_features = list(np.where(x_train.dtypes == 'object')[0])
#x_train, y_train = model_train[features], model_train.Survived
#x_test, y_test = model_test[features], model_test.Survived

#from catboost import CatBoostClassifier, Pool
#train_pool = Pool(x_train, y_train, cat_features)
#test_pool = Pool(x_test, y_test)
#clf = CatBoostClassifier(iterations = 200, verbose = False)
#clf.fit(train_pool)
#ypred = clf.predict(test_pool)
#model_test['pred'] = ypred.astype('int')

# Combine Scores
scores = pd.concat([rule[['PassengerId', 'pred']], model_test[['PassengerId', 'pred']]], axis = 0)
sub_new = pd.merge(scores, sub[['PassengerId']], on = 'PassengerId')
sub_new.columns = ['PassengerId', 'Survived']
sub_new = sub_new.sort_values(by = 'PassengerId')
sub_new.to_csv('submission.csv', index = False)

print(sub_new.Survived.sum(), sub_new.shape)


