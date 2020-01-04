"""
Decision Trees are crucial building blocks for more complex algorithms for classification & regression problems. 
They are based on the 'Binary Tree' data structure, which is a heirarchical and non-linear way of storing data. 

This code builds a very simple Decision Tree that only takes in binary inputs to classify a binary output.
Code contains two classes. 

A Node Class that contains:
-a partition of the data
-a left child node and a right child node
-a predicted probability for that partition of data
-functions to perform split on feature that increases purity of left/right partitions
-functions for visualisation of Nodes as Dict and Pydot Diagram

A Tree Class that contains:
-functions for recursive parition of training dataset through nodes linked to each other
-functions for recursive scoring of a new dataset using Decision Rules
-functions to combine scoring partitions and provide scores (probs/predictions)

Dataset: Adult Census Dataset that has been 'binarized' for simplicity, and contains 40k examples with 100 features.

"""

# Node Class
class Node:
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None
        self.prob = np.mean(self.data[:, 0:1])

    def entropy(self, data):
        ε = 0.00000001
        p = np.mean(data[:, 0:1])
        entropy = - (p * np.log(p + ε) + (1 - p) * np.log(1 - p + ε))
        return np.round(entropy, 4)

    def splitOn(self, index=1):
        idx = self.data[:, index:index + 1].reshape(self.data.shape[0])
        left = self.data[idx == 1, :]
        right = self.data[idx == 0, :]
        return left, right

    def WeightedEntropy(self, left, right):
        result = 0
        result += left.shape[0] * self.entropy(left[:, 0:1])
        result += right.shape[0] * self.entropy(right[:, 0:1])
        result = result / (left.shape[0] + right.shape[0])
        return result

    def bestSplitOn(self):
        self.initialEntropy = self.entropy(self.data[:, 0:1])
        self.bestSplitIdx = 1
        for i in range(1, self.data.shape[1]):
            left, right = self.splitOn(i)
            postSplitWeightedEntropy = self.WeightedEntropy(left, right)
            if postSplitWeightedEntropy <= self.initialEntropy:
                self.bestSplitIdx = i
        return self.bestSplitIdx

    def assignLeftRight(self, idx_for_split):
        left, right = self.splitOn(idx_for_split)
        self.left = Node(left)
        self.right = Node(right)

    def asDict(self):
        TreeAsDict = {}
        TreeAsDict['Examples'] = self.data.shape[0]
        TreeAsDict['Event-Rate'] = np.round(self.prob, 2)
        if self.left is not None:
            TreeAsDict['Split-Idx'] = self.bestSplitIdx

        if self.left is None:
            TreeAsDict['Left'] = None
        else:
            TreeAsDict['Left'] = self.left.asDict()

        if self.right is None:
            TreeAsDict['Right'] = None
        else:
            TreeAsDict['Right'] = self.right.asDict()
        return TreeAsDict

    def info(self, node):
        result = ''
        result += 'Shape: ' + str(node.data.shape) + '\n'
        result += 'Event Rate: ' + str(np.round(node.prob, 2)) + '\n'
        if node.left is not None:
            result += 'Split on: ' + str(node.bestSplitIdx) + '\n'
        return result

    def recursivePlotter(self, graph, node, level):
        if level > 0:
            edge = pydot.Edge(self.info(node), self.info(node.left))
            graph.add_edge(edge)
            edge = pydot.Edge(self.info(node), self.info(node.right))
            graph.add_edge(edge)
            graph = self.recursivePlotter(graph, node.left, level - 1)
            graph = self.recursivePlotter(graph, node.right, level - 1)
        return graph

    def asPlot(s, level):
        import pydot
        from PIL import Image
        graph = pydot.Dot(graph_type='graph')
        graph = s.recursivePlotter(graph, s, level=level)
        graph.write_png('DTgraph.png')
        im = Image.open("DTgraph.png")
        im.show()

# Tree Class
class Tree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def train(self, x_train, y_train):
        self.data = np.c_[y_train, x_train]
        self.root = Node(self.data)
        self.recursiveConstructor(self.root, self.max_depth)

    def recursiveConstructor(self, node, max_depth):
        if max_depth >= 0:
            node.assignLeftRight(node.bestSplitOn())
            self.recursiveConstructor(node.left, max_depth - 1)
            self.recursiveConstructor(node.right, max_depth - 1)

    def recursiveScore(self, node, max_depth, root):
        if (max_depth >= 0) & (root is not None):
            node.assignLeftRight(root.bestSplitIdx)
            node.prob = root.prob
            self.recursiveScore(node.left, max_depth - 1, root.left)
            self.recursiveScore(node.right, max_depth - 1, root.right)

    def combineScores(self, node, combineData, max_depth):
        if max_depth == 0:
            if combineData.size == 0:
                temp = np.c_[node.data, node.prob * np.ones((node.data.shape[0], 1))]
                combineData = np.c_[temp]
            else:
                temp = np.c_[node.data, node.prob * np.ones((node.data.shape[0], 1))]
                combineData = np.r_[combineData, temp]
        if max_depth > 0:
            combineData = self.combineScores(node.left, combineData, max_depth - 1)
            combineData = self.combineScores(node.right, combineData, max_depth - 1)
        return combineData

    def score(self, x_test, probs=False):
        x_test.shape
        self.scoring_data = np.c_[np.arange(x_test.shape[0]), x_test]
        self.scoring_root = Node(self.scoring_data)
        self.recursiveScore(self.scoring_root, self.max_depth, self.root)
        self.combineData = np.array([[]])
        self.combineData = self.combineScores(self.scoring_root, self.combineData, self.max_depth)
        self.combineData = self.combineData[model.combineData[:, 0].argsort()]
        self.probs = self.combineData[:, -1]
        self.preds = np.where(self.probs > 0.5, 1, 0)


# Evaluation
print('''Data is Adult US Census dataset that has been "binarized"''')

# Import packages & load data
import pandas as pd
import numpy as np
from pmlb import fetch_data
df = fetch_data('adult')
print(df.head())

# Dummify all categoricals
one_hot_encode = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
for i in one_hot_encode:
    df = pd.concat([df, pd.get_dummies(df[i], prefix=i, drop_first=True)], axis=1)
df.drop(one_hot_encode, axis=1, inplace=True)

# Binarize all the continous variables
df['age_dummy'] = 0
df.loc[df['age'] > 38, 'age_dummy'] = 1

df['fnlwgt_dummy'] = 0
df.loc[df['fnlwgt'] > 189664, 'fnlwgt_dummy'] = 1

df['education-num_dummy'] = 0
df.loc[df['education-num'] > 10, 'education-num_dummy'] = 1

df['capital-gain_dummy'] = 0
df.loc[df['capital-gain'] > 1079, 'capital-gain_dummy'] = 1

df['capital-loss_dummy'] = 0
df.loc[df['capital-loss'] > 87, 'capital-loss_dummy'] = 1

df['hours-per-week_dummy'] = 0
df.loc[df['hours-per-week'] > 40, 'hours-per-week_dummy'] = 1

# Drop continous columns & Numpyfy
cols_to_drop = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
df.drop(cols_to_drop, axis=1, inplace=True)
feature_names = list(df.columns)
df = df.sample(frac=1)
data = np.array(df)

# Train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data[:, 1:], data[:, 0:1], test_size=0.3, random_state=3)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# Train the model
model = Tree(max_depth=3)
model.train(x_train, y_train)

# Visualise the Tree 1: Dict
TreeAsDict = model.root.asDict()
import pprint
pprint.pprint(TreeAsDict)

# Visualise the Tree 2: Graphviz
import pydot
model.root.asPlot(level=3)

# Score and Evaluate
from sklearn.metrics import roc_auc_score, accuracy_score
model.score(x_test)
print('\n Evaluation: ')
print('\tTest AUC: ', roc_auc_score(y_test, model.probs))
print('\tTest Accuracy: ', accuracy_score(y_test, model.preds))
