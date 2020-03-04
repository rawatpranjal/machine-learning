test_path = '/axp/rim/imsadsml/dev/ppranja/kaggle/titanic/test.csv'
train_path = '/axp/rim/imsadsml/dev/ppranja/kaggle/titanic/train.csv'
sub_path = '/axp/rim/imsadsml/dev/ppranja/kaggle/titanic/gender_submission.csv'
new_sub_path = '/axp/rim/imsadsml/dev/ppranja/kaggle/titanic/submission.csv'

import pandas as pd, numpy as np
sub = pd.read_csv(sub_path)
test = pd.read_csv(test_path)
train = pd.read_csv(train_path)

df_temp = train.copy()
df_temp['Surname'] = df_temp.Name.str.split(',').str[0]
nameSurv = pd.concat([df_temp[['Surname', 'Survived']].groupby('Surname').mean(), df_temp[['Surname', 'Survived']].groupby('Surname').count()], axis = 1)
nameSurv.reset_index(inplace = True)
nameSurv.columns = ['Surname', 'Mean', 'Count']
surname_survived = list(nameSurv[nameSurv.Mean == 1][nameSurv.Count > 1].Surname)
df_temp['surname_survival'] = 0
df_temp.loc[df_temp['Surname'].isin(surname_survived), 'surname_survival'] = 1
print(df_temp[['surname_survival', 'Survived']].groupby('surname_survival').mean())


def pipeline(df, surname_survived):
    df['Gender'] = 0
    df.loc[df.Sex == 'male', 'Gender'] = 1

    df['Embarked_S'] = 0
    df.loc[df.Embarked == 'S', 'Embarked_S'] = 1

    df['Embarked_C'] = 0
    df.loc[df.Embarked == 'C', 'Embarked_C'] = 1

    df['family'] = df['Parch'] + df['SibSp'] + 1
    
    df['Surname'] = df.Name.str.split(',').str[0]
    df['surname_survival'] = 0
    df.loc[df['Surname'].isin(surname_survived), 'surname_survival'] = 1
    
    df = df.replace('', np.nan)
    age_mode = df['Age'][df.Age.notna()].mode()
    df['Age'] = df['Age'].fillna(age_mode[0])
    
    fare_mode = df['Fare'].mode()
    df['Fare'] = df['Fare'].fillna(fare_mode[0])
    
    df['Unmarried_woman'] = 0
    df.loc[df.Name.str.contains('Miss'), 'Unmarried_woman'] = 1

    df['Child'] = 0
    df.loc[df.Name.str.contains('Master'), 'Child'] = 1

    cols_to_drop = [ 'Name', 'Sex', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Embarked', 'Surname']
    df.drop(cols_to_drop, axis = 1, inplace = True)
    return df

train = pipeline(train, surname_survived)
test = pipeline(test, surname_survived)

def featureScale(df):
    e = 0.0000000001
    return (df - df.mean(axis = 0))/ df.std(axis = 0)

features = list(train.columns)
features.remove('Survived')
features.remove('PassengerId')
train[features] = featureScale(train[features])
test[features] = featureScale(test[features])

import torch as t
t.manual_seed(4)
y = t.tensor(train[['Survived']].values, dtype=t.float)
x = t.tensor(train[features].values, dtype=t.float)

train_size = 0.7
training_rows = t.LongTensor(round(x.shape[0]*train_size)).random_(0, x.shape[0])
validation_rows = [i for i in list(range(x.shape[0])) if i not in training_rows]
x_train, y_train = x[training_rows], y[training_rows]
x_val, y_val = x[validation_rows], y[validation_rows]

print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

n_input, n_output, hidden = x_train.shape[1], y_train.shape[1], x_train.shape[1] + 5
model = t.nn.Sequential(
    t.nn.Linear(n_input, hidden),
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
loss_fn = t.nn.BCELoss(reduction='mean')
optimizer = t.optim.Adam(model.parameters(), lr=0.1, weight_decay = 0.01)

from sklearn.metrics import accuracy_score
def eval_metric(a, b):
    pred = t.round(a)
    return round(accuracy_score(pred.detach().numpy(), b.detach().numpy()), 2)

# Train
for i in range(1000):
    loss = loss_fn(model(x_train), y_train)
    if i % 100 == 0:
        print(i, round(loss.item(), 2), eval_metric(model(x_train), y_train), eval_metric(model(x_val), y_val))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
scores = model(t.tensor(test[features].values, dtype=t.float))
scores = t.round(scores)
sub['Survived'] = scores.int().detach().numpy()
sub.to_csv(new_sub_path, index=False)
    
