"""
Decision Trees are crucial building blocks for more complex algorithms for classification & regression problems. 
They are based on the 'Binary Tree' data structure, which is a heirarchical and non-linear way of storing data. 

This code builds a very simple Decision Tree that only takes in binary inputs to classify a binary output.
Code contains two classes. 

A Node Class that contains:
-a partition of the data
-a left child node and a right child node
-a predicted probability for that partition of data, and other statistics from the data contained in it
-functions to perform split on feature that increases purity of left/right partitions (measured by Gini)
-functions for visualisation of Nodes as Pydot Diagram

A Binary Tree Class that contains:
-functions for recursive paritioning of training dataset through nodes linked to each other i.e "TRAINING"
-functions for recursive scoring of a new dataset using Decision Rules in trained tree i.e "SCORING"
-functions to combine scoring partitions and provide scores (probs/predictions) 

Dataset: "Binarized" Breast Cancer Dataset from SKLEARN

"""
# Core Component
class Node:
    def __init__(self, data):
        self.data = data
        self.y = data[:, 0:1]
        self.x = data[:, 1:data.shape[0] - 3]
        self.m = data.shape[0]
        self.n = data.shape[1]
        self.p = np.mean(data[:, 0:1])
        self.q = 1 - np.mean(data[:, 0:1])
        self.weights = data[:, -1:]
        self.decision = np.where(self.p>0.5, 1, 0).item()
        self.prediction = data[:, -3:]
        self.idx = data[:, -3:-2]
        self.minChildSize = 5
        self.bestSplitIdxFinder()
        self.left_data, self.right_data = self.createLeftRightData()
        self.left, self.right = None, None

    def gini(self, data):
        p = np.mean(data[:, 0])
        q = 1 - p
        gini = 1 - p ** 2 - q ** 2
        return gini

    def split(self, idx):
        df = self.data
        left, right = df[df[:, idx]==1], df[df[:, idx]==0]
        return left, right

    def weightedGini(self, left, right):
        weightedGini = left.shape[0] * self.gini(left) + right.shape[0] * self.gini(right)
        weightedGini = weightedGini/(left.shape[0]+right.shape[0])
        return weightedGini

    def bestSplitIdxFinder(self):
        initGini = self.gini(self.data)
        self.bestSplitIdx = 1
        for i in range(1, self.n - 3):
            left, right = self.split(i)
            if ((left.shape[0]>self.minChildSize) & (right.shape[0]>self.minChildSize)):
                weightedGini = self.weightedGini(left, right)
                if weightedGini < initGini:
                    self.bestSplitIdx = i

    def createLeftRightData(self):
        left, right = self.split(self.bestSplitIdx)
        if ((left.shape[0]>self.minChildSize) & (right.shape[0]>self.minChildSize)):
            left_pred = np.mean(left[:, 0])*np.ones(left.shape[0])
            right_pred = np.mean(right[:, 0])*np.ones(right.shape[0])
            left[:, -2:-1] = left_pred.reshape(-1, 1)
            right[:, -2:-1] = right_pred.reshape(-1, 1)
            return left, right
        else:
            return None, None

    def assign(self):
        if self.left_data is not None:
            self.left = Node(self.left_data)
            self.right = Node(self.right_data)
        else:
            pass

    def recursiveCreator(self, depth, minChildSize = 5):
        if (depth >0):
            self.minChildSize = minChildSize
            self.assign()
            if (self.left_data is not None):
                self.left.recursiveCreator(depth-1)
                self.right.recursiveCreator(depth-1)

    def info(self):
        result = 'Shape: ' + str(self.data.shape) + '\n'
        result += 'Split On: ' + str(self.bestSplitIdx) + '\n'
        result += 'Prediction: ' + str(self.decision) + '\n'
        result += 'Probability: ' + str(round(self.p, 2))
        return result

    def recursiveScorer(self, depth, node):
        if ((depth >= 0) & (node is not None)):

            self.bestSplitIdx = node.bestSplitIdx
            self.left_data, self.right_data = self.split(self.bestSplitIdx)
            self.prediction[:, 1] = node.p
            self.p = node.p
            self.decision = np.where(node.p>0.5, 1, 0)

            if ((self.left_data.shape[0]>0) & (self.right_data.shape[0]>0)):
                self.left, self.right = Node(self.left_data), Node(self.right_data)
            else:
                self.left, self.right = None, None

            if node.left is not None:
                self.left.recursiveScorer(depth -1, node.left)
                self.right.recursiveScorer(depth -1, node.right)

    def recursivePlot(self, graph, depth):
        if ((depth>0) & (self.left is not None)):
            edge = pydot.Edge(self.info(), self.left.info())
            graph.add_edge(edge)
            edge = pydot.Edge(self.info(), self.right.info())
            graph.add_edge(edge)
            graph = self.left.recursivePlot(graph, depth-1)
            graph = self.right.recursivePlot(graph, depth-1)
        return graph

    def obtainScores(self, depth, scores = None):
        if (depth==0):
            if scores is None:
                scores = self.prediction
            else:
                scores = np.r_[scores, self.prediction]

        if self.left is not None:
            scores=self.left.obtainScores(depth-1, scores)
            scores=self.right.obtainScores(depth-1, scores)
        scores = scores[scores[:, 0].argsort()]
        return scores

    def asPlot(self, depth):
        import pydot
        from PIL import Image
        graph = pydot.Dot(graph_type='graph')
        graph = self.recursivePlot(graph, depth)
        graph.write_png('DTgraph.png')
        im = Image.open("DTgraph.png")
        im.show()

# Model
class BinaryDecisionTree:
    def __init__(self, depth, minChildSize = 5):
        self.depth = depth
        self.minChildSize = minChildSize

    def dataProcess(self, y_train, x_train):
        if y_train is None:
            y_train = np.ones(x_train.shape[0])
        m = y_train.shape[0]
        initPred = np.mean(y_train)*np.ones(y_train.shape[0])
        initWeights = np.ones(y_train.shape[0])/y_train.shape[0]
        data = np.c_[y_train, x_train, np.arange(m), initPred, initWeights]
        np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        return data

    def train(self, x_train, y_train, eval_set = None, visualize = True):
        training_data = self.dataProcess(y_train, x_train)
        training_root = Node(training_data)
        training_root.recursiveCreator(self.depth, self.minChildSize)

        print('Training Binary Decision Tree...')
        print(f'\tDepth: {self.depth}')
        print(f'\tMin Child Size: {self.minChildSize}')
        from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
        print(f'\tTrain AUC: {roc_auc_score(y_train, training_root.obtainScores(self.depth)[:, 1]):0.2}')
        print(f'\tTrain LogLoss: {roc_auc_score(y_train, training_root.obtainScores(self.depth)[:, 1]):0.2}')
        print(f'\tTrain Accuracy: {accuracy_score(y_train, np.where(training_root.obtainScores(self.depth)[:, 1]>0.5, 1, 0)):0.2}')

        if eval_set is not None:
            scoring_data = self.dataProcess(None, x_test)
            scoring_root = Node(scoring_data)
            scoring_root.recursiveScorer(self.depth, training_root)
            self.scoring_root = scoring_root
            print(f'\n\tTest AUC: {roc_auc_score(y_test, scoring_root.obtainScores(self.depth)[:, 1]):0.2}')
            print(f'\tTest LogLoss: {roc_auc_score(y_test, scoring_root.obtainScores(self.depth)[:, 1]):0.2}')
            print(f'\tTest Accuracy: {accuracy_score(y_test, np.where(scoring_root.obtainScores(self.depth)[:, 1]>0.5, 1, 0)):0.2}')
        if visualize == True:
            import pydot
            training_root.asPlot(self.depth)
            if eval_set is not None:
                scoring_root.asPlot(self.depth)
        self.training_root = training_root

    def score(self, x_test):
            scoring_data = self.dataProcess(None, x_test)
            scoring_root = Node(scoring_data)
            scoring_root.recursiveScorer(self.depth, self.training_root)
            return scoring_root.obtainScores(self.depth)[:, 1]


# Evaluation
import numpy as np
import pydot
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

from sklearn.datasets import load_breast_cancer
x, y = load_breast_cancer(return_X_y = True)
x = np.where(x > x.mean(axis = 0), 1, 0)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

model = BinaryDecisionTree(depth = 3)
model.train(x_train, y_train, eval_set = [x_test, y_test], visualize = True)
scores = model.score(x_test)

print('Scores')
print(scores[0:10])
