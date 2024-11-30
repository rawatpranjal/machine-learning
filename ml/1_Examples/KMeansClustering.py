'''
Core aspects of Andrew Ng's Machine Learning Coursera Course
Topic: Week 8: K Means Clustering
Data: Iris Data (from the Coursera Course)
Problem: Clustering
Features = 4
Observations = 150k, 50 of different class.
'''

PATH = '/Users/pranjal/Google Drive/python_projects/projects/courses/andrew_ng/machine_learning/algorithms/KmeansClustering/iris.data'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as distance


def importData(PATH, sample_size):
    '''Import the data or a sample of it'''
    df = pd.read_table(PATH, sep=',', names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'])
    df = df.sample(frac=1)
    examples = round(sample_size * df.shape[0])
    df = df.iloc[0:examples, :]
    df['class'] = 0
    df.loc[df['Species'] == 'Iris-setosa', 'class'] = 1
    df.loc[df['Species'] == 'Iris-versicolor', 'class'] = 2
    y = df['class']
    y = np.array(y)
    df = df.drop(['Species', 'class'], axis=1)
    X = np.array(df)
    return X, y


def featureScale(X):
    '''Mean Normalization & Min-Max Scaling'''
    temp0 = X - X.mean(axis=0)
    temp1 = X.max(axis=0) - X.min(axis=0)
    return np.round(np.divide(temp0, temp1 + 0.00000001), 2)


def initCentroids(X, K, rdm):
    '''Randomly select K centroids from the data'''
    np.random.seed(rdm)
    m, n = X.shape[0], X.shape[1]
    idx = np.random.choice(m, K, replace=False)
    centroids = X[idx, :]
    return centroids


def assignClusters(X, centroids):
    '''Assign Clusters to examples by Distance to Centroids'''
    m, n = X.shape[0], X.shape[1]
    no_k = centroids.shape[0]
    distances = np.empty((m, no_k))
    for i in range(no_k):
        temp = sum(np.power(X - centroids[i, :], 2).T)
        distances[:, i] = temp
    assignment = np.argmin(distances, axis=1).reshape(m, 1)
    return assignment, distances


def costFunction(assignment, distances):
    '''Distortion/Cost Function for K-Means Clustering'''
    centroidIndices = np.arange(distances.shape[0])[:, None]
    leastDist = distances[centroidIndices, assignment]
    leastDist = leastDist / leastDist.shape[0]
    return sum(sum(leastDist))


def updateCentroids(X, centroids, assignment):
    '''Update Centroids to Mean of Cluster'''
    m, n = X.shape[0], X.shape[1]
    chosen_centroids = list(set(list(i[0] for i in list(assignment))))
    centroids = np.empty((len(chosen_centroids), X.shape[1]))
    temp = np.c_[assignment, X]
    cnt = 0
    for i in chosen_centroids:
        subset = temp[temp[:, 0] == i][:, 1:n + 1]
        update = sum(subset) / subset.shape[0]
        update = update.reshape(1, subset.shape[1])
        centroids[cnt, :] = update.reshape(1, subset.shape[1])
        cnt += 1
    return centroids


def KmeansIter(X, K, random_state):
    '''Randomly initialize and run Kmeans'''
    centroids = initCentroids(X, K, random_state)
    cost_path = []
    cost_old, cost_new = 0, 1
    while cost_old != cost_new:
        cost_old = cost_new
        assignment, distances = assignClusters(X, centroids)
        cost_new = costFunction(assignment, distances)
        cost_path.append(cost_new)
        centroids = updateCentroids(X, centroids, assignment)
    return centroids, cost_path, assignment


def KmeansClustering(X, K, random_state, inits):
    '''Best Kmeans Result is selected from multiple inits'''
    print('\n RUNNING K MEANS CLUSTERING ALGORITHM....\n')
    np.random.seed(random_state)
    randomness = np.random.choice([i for i in range(inits)], (inits, ))
    lowCostPath = []
    centroids_path = []
    assignment_path = []
    for i in range(inits):
        centroids, cost_path, assignment = KmeansIter(X, K, randomness[i])
        lowCostPath.append([cost_path[-1], centroids.shape[0]])
        centroids_path.append(centroids)
        print(f'Random Initialization {i}, Cost = {cost_path[-1]}')
    lowestCostIdx = lowCostPath.index(min(lowCostPath))
    print(f'Selected Init {lowestCostIdx}, Cost = {lowCostPath[lowestCostIdx][0]}')
    X_clustered = np.c_[assignment, X]
    return centroids_path[lowestCostIdx], X_clustered, lowCostPath[lowestCostIdx][0]


def internalValidation(X_clustered):
    '''Silouhette Coefficient'''
    allClusters = []
    for i in set(X_clustered[:, 0]):
        temp = X_clustered[X_clustered[:, 0] == i]
        allClusters.append(temp)

    silouhette_values = []
    for i in allClusters:
        for j in i:
            intraDist = distance.cdist(j.reshape((1, i.shape[1])), i.reshape((i.shape[0], i.shape[1])), 'euclidean')
            avgIntraDist = np.sum(intraDist) / 2
            temp = np.mean(intraDist)

            lowestAvgInterDist = 10000000
            for r in allClusters:
                interDist = distance.cdist(j.reshape((1, i.shape[1])), r.reshape((r.shape[0], r.shape[1])), 'euclidean')
                avgInterDist_r = np.mean(interDist)
                if (avgInterDist_r < lowestAvgInterDist and avgInterDist_r != temp):
                    avgInterDist = avgInterDist_r
                    lowestAvgInterDist = avgInterDist

            if i.shape[0] == 1:
                silouhette_i = 0
            else:
                if avgIntraDist <= lowestAvgInterDist:
                    silouhette_i = 1 - avgIntraDist / lowestAvgInterDist
                else:
                    silouhette_i = 1 - lowestAvgInterDist / avgIntraDist
            silouhette_values.append(silouhette_i)
    return np.round(np.mean(silouhette_values), 4)


def externalValidation(X_clustered, y):
    '''Multi-Class Accuracy'''
    import scipy.stats
    predicted = np.round(X_clustered[:, 0]).astype('int')
    predicted = predicted + 1
    y_temp = y.reshape(X_clustered.shape[0])
    y_temp = y_temp + 1
    predicted2 = predicted.copy()
    temp = np.c_[predicted, y_temp]
    for i in list(set(predicted)):
        mode = scipy.stats.mode(temp[temp[:, 0] == i][:, 1])[0][0]
        predicted2 = np.where(predicted2 == i, - mode, predicted2)
    predicted2 = predicted2 * -1
    accuracy = np.where(predicted2 == y_temp, 1, 0)
    return round(sum(accuracy) / len(accuracy), 4)


def elbowCurve(K_list, X, y, random_state, inits):
    for i in K_list:
        print('\nIterating with {} Clusters...'.format(i))
        centroids, X_clustered, cost = KmeansClustering(X, i, random_state, inits)
        plt.subplot(1, 2, 1)
        plt.scatter(i, internalValidation(X_clustered), c='r')

        plt.subplot(1, 2, 2)
        plt.scatter(i, cost, c='g')
    plt.subplot(1, 2, 1).set_title('Silhouette Plot')
    plt.subplot(1, 2, 2).set_title('Elbow Curve')
    plt.show()


'''
Code Run
'''
X, y = importData(PATH, 1)
print('Data Shape: ', X.shape, y.shape, '\n')
X = featureScale(X)
K = 3
random_state = 42
inits = 100
centroids, X_clustered, cost = KmeansClustering(X, K, random_state, inits)
print(f'Chosen Centroids: \n {centroids}')
print(f'\nInternal Validation - Silhouette: {internalValidation(X_clustered)}')
print(f'\nExtenal Validation - MultiClass Accuracy: {externalValidation(X_clustered, y)}')
print('\n DIAGNOSIS: SILHOUETTE PLOT & ELBOW CURVE \n')
K_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
elbowCurve(K_list, X, y, random_state, 5)
