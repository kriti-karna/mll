import numpy as np
import pandas as pd
from copy import deepcopy


def euclidean(a,b, ax=1):
    return np.linalg.norm(a-b, axis=ax)



k = 3
data = pd.read_csv('kmeans.csv')
print(data)

x1 = data['X1'].values
x2 = data['X2'].values
X = np.array(list(zip(x1, x2)))
print(X)
C_x = [6.2, 6.6 ,6.5]
C_y = [3.2, 3.7, 3.0]
Centroid = np.array(list(zip(C_x, C_y)))
print("Initial Centroids")
print(Centroid.shape)

Centroid_old = np.zeros(Centroid.shape)
print(Centroid_old)

clusters = np.zeros(len(X))
print("clusters: ",clusters)
error = euclidean(Centroid, Centroid_old, None)
print(error)
iterr = 0

while error != 0:

    iterr = iterr + 1
    for i in range(len(X)):

        distances = euclidean(X[i], Centroid)
        cluster = np.argmin(distances)
        clusters[i] = cluster

    Centroid_old = deepcopy(Centroid)

    for p in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == p]
        Centroid[p] = np.mean(points, axis=0)
    print("Centre of the clusters after ", iterr," Iteration \n", Centroid)
    error = euclidean(Centroid, Centroid_old, None)
    print("Error  ... ",error)
