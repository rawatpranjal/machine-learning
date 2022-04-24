"""
Decision Trees are crucial building blocks for more complex algorithms for classification & regression problems. 
They are based on the 'Binary Tree' data structure, which is a heirarchical and non-linear way of storing data.
Nodes carry information and link to child nodes. Tree traversal is done by means of recursive functions that act 
on nodes and thier children repeatedly until some terminal condition is met. 

This code builds a very simple Decision Tree that takes a dataset and recursively partitions on features to segment
the labels in its dataset. The tree can be of arbitrary depth but only takes in binary inputs when segmenting.


A Node Class that contains:
-a partition of the data
-a left child node and a right child node
-a predicted probability for that partition of data, and other statistics from the data contained in it
-functions to perform split on feature that increases purity of left/right partitions (measured by Gini)
-functions for visualisation of Nodes as Pydot Diagram 
-functions for recursive paritioning of training dataset through nodes linked to each other i.e "TRAINING"
-functions for recursive scoring of a new dataset using Decision Rules in trained tree i.e "SCORING TREE"
-functions to combine scoring partitions and provide scores (probs/predictions) "SCORING"

Additional:
Data is a "Binarized" Synthetic Dataset. Care has been taken to allow training tree to completely overfit the training
data, i.e as many splits as possible can be made. While scoring a fresh dataset, the new tree will split as many times
as it can following the decision rules from the trained tree. 
"""

import numpy as np
np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})
import pandas as pd
import pydot

# Building block for the tree


class Node:
    def __init__(self, data):
        self.data = data
        self.m = data.shape[0]
        self.n = data.shape[1]
        self.y = data[:, 0:1]
        self.x = data[:, 1:self.n - 3]
        self.p = np.mean(self.y)
        self.gini_val = self.impurity(self.data)
        self.q = 1 - self.p
        self.decision = int(round(self.p))
        self.scores = data[:, -3:]
        self.weights = data[:, -1:]
        self.probs = data[:, -2:-1]
        self.idx = data[:, -3:-2]
        self.bestSplitIdxFinder()
        self.left_data, self.right_data = self.createLeftRightData()
        self.left, self.right = None, None
        self.leaf = True
        self.ε = None
        self.α = None

    # Gini measure of purity/impurity
    def impurity(self, data):
        e = 0.00000001
        weights = data[:, -1:]
        p = np.mean(data[:, 0] * weights)
        q = 1 - p
        impurity = 1 - p * p - q * q
        return impurity

    # Splits data on feature index
    def split(self, idx):
        df = self.data
        left, right = df[df[:, idx] == 0], df[df[:, idx] == 1]
        return left, right

    # Calculates weighted impurity in left and right datasets
    def weightedImpurity(self, left, right):
        weightedImpurity = left.shape[0] * self.impurity(left) / (left.shape[0] + right.shape[0])
        weightedImpurity += right.shape[0] * self.impurity(right) / (left.shape[0] + right.shape[0])
        return weightedImpurity

    # Selects optimal feature index that creates largest reduction in impurity
    def bestSplitIdxFinder(self):
        base = self.impurity(self.data)
        self.bestSplitIdx = 1
        for i in range(1, self.n - 3):
            left, right = self.split(i)
            if ((left.shape[0] > 0) & (right.shape[0] > 0)):
                weightedImpurity = self.weightedImpurity(left, right)
                if weightedImpurity < base:
                    self.bestSplitIdx = i
                    base = weightedImpurity

    # Basis optimal split index, splits dataset into left and right
    def createLeftRightData(self):
        left, right = self.split(self.bestSplitIdx)
        if ((left.shape[0] > 0) & (right.shape[0] > 0)):
            left_pred = np.mean(left[:, 0]) * np.ones(left.shape[0])
            right_pred = np.mean(right[:, 0]) * np.ones(right.shape[0])
            left[:, -2:-1] = left_pred.reshape(-1, 1)
            right[:, -2:-1] = right_pred.reshape(-1, 1)
            return left, right
        else:
            return None, None

    # Assigns left/right child nodes using left/right datasets
    # Parent node is not considered a 'leaf' node anymore
    def assign(self):
        if self.left_data is not None:
            self.left = Node(self.left_data)
            self.right = Node(self.right_data)
            self.leaf = False

    # Recursively assigns nodes until terminal condition is met
    # From root node, builds tree of a given depth
    def recursiveCreator(self, depth):
        if (depth > 0):
            self.assign()
            if (self.left_data is not None):
                self.left.recursiveCreator(depth - 1)
                self.right.recursiveCreator(depth - 1)

    # Compiles key information about given node
    # Unique ID to enable clean tree visuals in Pydot
    def info(self, depth):
        result = 'Shape: ' + str(self.data.shape[0]) + '\n'
        result += 'Impurity: ' + str(round(self.gini_val, 3)) + '\n'
        result += 'Split On: ' + str(self.bestSplitIdx - 1) + '\n'
        result += 'Prediction: ' + str(self.decision) + '\n'
        result += 'Probability: ' + str(round(self.p, 2)) + '\n'
        result += 'ID:' + str(depth) + str(int(self.data[0, -3]))
        return result

    # Recursively plots parent and child nodes until it cannot
    def recursivePlot(self, graph, depth):
        if ((depth > 0) & (self.left is not None) & (self.right is not None)):
            edge = pydot.Edge(self.info(depth), self.left.info(depth - 1))
            graph.add_edge(edge)
            edge = pydot.Edge(self.info(depth), self.right.info(depth - 1))
            graph.add_edge(edge)
            graph = self.left.recursivePlot(graph, depth - 1)
            graph = self.right.recursivePlot(graph, depth - 1)
        return graph

    # imports packages and from root, plots entire tree
    def asPlot(self, depth):
        import pydot
        from PIL import Image
        graph = pydot.Dot(graph_type='graph')
        graph = self.recursivePlot(graph, depth)
        graph.write_png('DTgraph.png')
        im = Image.open("DTgraph.png")
        im.show()

    # Uses spliting rules from an existing tree ("trained tree"), to create another tree ("scored tree") from a fresh node
    # Makes two checks at every node -> is training tree node a leaf? is it possible to split scoring tree node further?
    # if both checks are cleared, scoring node is split using optimal feature split rule from the training node
    def recursiveScorer(self, depth, node):
        if node.leaf == True:
            self.scores[:, 1] = node.p
            self.p = node.p
            self.decision = node.decision
            self.leaf = True

        elif node.leaf == False:
            self.scores[:, 1] = node.p
            self.p = node.p
            self.decision = node.decision

            self.bestSplitIdx = node.bestSplitIdx
            self.left_data, self.right_data = self.split(self.bestSplitIdx)
            if ((self.left_data.shape[0] > 0) & (self.right_data.shape[0] > 0)):
                self.left = Node(self.left_data)
                self.right = Node(self.right_data)
                self.leaf = False

                self.left.recursiveScorer(depth - 1, node.left)
                self.right.recursiveScorer(depth - 1, node.right)

    # Picks up 'scores' i.e example index, prob prediction & example weight from all 'leaf' nodes
    # Appends them all and yeilds the final 'scores' of a tree
    def obtainScores(self, depth, scores=None):
        if self.leaf == False:
            if self.left is not None:
                scores = self.left.obtainScores(depth - 1, scores)
            if self.right is not None:
                scores = self.right.obtainScores(depth - 1, scores)
        else:
            if scores is None:
                scores = self.scores
            else:
                scores = np.r_[scores, self.scores]
        scores = scores[scores[:, 0].argsort()]
        return scores


# Synthetic Dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
x, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_clusters_per_class=3, random_state=42)
x = np.where(x > np.mean(x, axis=0), 1, 0)  # suitably binarized features
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Combines labels & features; adds (i) row indices, (ii) prob predictions & (iii) example weights
# This is the base data on which we build the tree


def dataProcess(y_train, x_train):
    if y_train is None:
        y_train = np.ones(x_train.shape[0])
    m = y_train.shape[0]
    initPred = np.mean(y_train) * np.ones(m)
    initWeights = np.ones(m)
    data = np.c_[y_train, x_train, np.arange(m), initPred, initWeights]
    return data


data = dataProcess(y_train, x_train)
print(data)

# Train the tree
depth = 5
print('Depth: ', depth)
print('Binary Tree on Training Data')
training_root = Node(data)
training_root.recursiveCreator(depth)

# Visualise
training_root.asPlot(depth)

# Evaluate
probs = training_root.obtainScores(depth)[:, 1]
preds = np.where(probs > 0.5, 1, 0)
print('\tAUC', roc_auc_score(y_train, probs))
print('\tLogloss', log_loss(y_train, probs))
print('\tAccuracy', accuracy_score(y_train, preds))

# Score Test Data
print('Binary Tree on Test Data')
scoring_data = dataProcess(y_test, x_test)
scoring_root = Node(scoring_data)
scoring_root.recursiveScorer(depth, training_root)

# Visualise
scoring_root.asPlot(depth)

probs = scoring_root.obtainScores(depth)[:, 1]
preds = np.where(probs > 0.5, 1, 0)
print('\tAUC', roc_auc_score(y_test, probs))
print('\tLogloss', log_loss(y_test, probs))
print('\tAccuracy', accuracy_score(y_test, preds))

# Function to count leaves


def leafCount(node, cnt=0):
    if node.leaf == True:
        cnt += 1
    else:
        cnt = leafCount(node.left, cnt)
        cnt = leafCount(node.right, cnt)
    return cnt


print('Leaves in Training Tree: ', leafCount(training_root))
print('Leaves in Scoring Tree: ', leafCount(scoring_root))

# Comparing with SKlearn's packages
print('Benchmark to Sklearn')
print('Depth: ', depth)

from sklearn import tree
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
clf = tree.DecisionTreeClassifier(max_depth=depth, random_state=42, criterion='gini')
clf = clf.fit(x_train, y_train)

print('Training Data')
probs = clf.predict_proba(x_train)
preds = clf.predict(x_train)
print('\tAUC', roc_auc_score(y_train, probs[:, 1]))
print('\tLogloss', log_loss(y_train, probs[:, 1]))
print('\tAccuracy', accuracy_score(y_train, preds))

# Visualize Sklearn's tree
import graphviz
dot_data = tree.export_graphviz(clf, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("test")
dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.view()

print('Test Data')
probs = clf.predict_proba(x_test)
preds = clf.predict(x_test)
print('\tAUC', roc_auc_score(y_test, probs[:, 1]))
print('\tLogloss', log_loss(y_test, probs[:, 1]))
print('\tAccuracy', accuracy_score(y_test, preds))

