'''
GRAPH EMBEDDINGS - V1

Text and images are often converted into vector "embeddings" to process them in ML algorithms. 
The same can be done for graphs.

This works on a list of source and destination nodes, called edges. 
The edges provided become "positive edges" from which counter examples or "negative edges" are created.

A Comparator and Operator function takes the embeddings of any two nodes and produces a score. 
This score contributes to a global loss function. There may be regularization.

Positive edges and negative edges contribute differently to the global loss. 
They might be passed through a softmax layer as well.

The final output are graph embeddings and comparator scores that might be used directly (for recommendations, predictions) 
or indirecty (as features in downstream classifiers).

This is a semi-supervised process, as we only obtain positive examples and negative examples are being generated 
to create contrast.
'''

edgelist = '''a,1
a,2
a,3
a,4
b,3
b,4
b,5
b,6
c,5
c,6
c,7
c,8
d,7
d,8
d,1
d,2'''


def preprocessing(edgelist):
    edges = [i.split(',') for i in edgelist.split('\n')]
    lhs = set([i.split(',')[0] for i in edgelist.split('\n')])
    rhs = set([i.split(',')[1] for i in edgelist.split('\n')])
    nodes = lhs.union(rhs)
    return edges, nodes, lhs, rhs


def negativeSampling(edges, lhs, rhs):
    edges_neg = []
    for i in lhs:
        pos = [j[1] for j in edges if j[0] == i]
        neg = []
        for j in rhs:
            if j not in pos:
                neg.append(j)
        for j in neg:
            edges_neg.append([i, j])
    return edges_neg


def initalize(nodes, D=25):
    from torch import randn
    node2vec = {i: randn(D, 1, requires_grad=True) for i in nodes}
    W = randn(D, D, requires_grad=True)
    b = zeros(D, 1, requires_grad=True)
    return node2vec, W, b


def comparator(vec1, vec2, W, b):
    left = vec1
    right = mm(W, vec2) + b  # asymmetry
    score = CosineSimilarity(dim=0)(left, right)
    return score


def loss_function(edges, edges_neg):
    N = len(edges) + len(edges_neg)
    cost = 0
    for i in edges_neg:
        score = comparator(node2vec[i[0]], node2vec[i[1]], W, b)
        prob = Sigmoid()(score)
        cost -= log(1 - prob) / N
    for i in edges:
        score = comparator(node2vec[i[0]], node2vec[i[1]], W, b)
        prob = Sigmoid()(score)
        cost -= log(prob) / N
    return cost


def train(edges, edges_neg, parameters, epochs=100, lr=0.01, verbose=True):
    from torch.optim import Adam
    optimizer = Adam(list(node2vec.values()) + [W, b], lr=lr)
    print('Training...')
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_function(edges, edges_neg)
        loss.backward()
        optimizer.step()
        if verbose == True:
            if epoch % round(epochs / 10) == 0:
                print(f'{epoch}: {loss.item():0.2f}')


# Training
import torch as t
from torch import randn, zeros, mm, log, norm, dist
from torch.nn import CosineSimilarity
from torch.nn import Sigmoid
edges, nodes, lhs, rhs = preprocessing(edgelist)
edges_neg = negativeSampling(edges, lhs, rhs)
node2vec, W, b = initalize(nodes, D=10)
parameters = list(node2vec.values()) + [W, b]
train(edges, edges_neg, parameters)

# Evaluate Comparator Scores
print('\nScores for Negative Edges')
for i in edges_neg:
    print(i[0], i[1], round(comparator(node2vec[i[0]], node2vec[i[1]], W, b).item(), 2))

print('\nScores for Positive Edges')
for i in edges:
    print(i[0], i[1], round(comparator(node2vec[i[0]], node2vec[i[1]], W, b).item(), 2))

print('\nNode Embeddings:')
for i in nodes:
    print(i, node2vec[i])
    print('\n')
