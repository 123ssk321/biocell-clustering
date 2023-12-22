import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from queue import PriorityQueue
from itertools import count


class HierarchicalKMeans(BaseEstimator):
    def __init__(self, max_iter=7):
        self.max_iter = max_iter
        self.kmeans = KMeans(n_clusters=2)

    def fit(self, X):
        return self

    def predict(self, X):
        tiebreaker = count()
        clusters = PriorityQueue()
        output = [[] for _ in range(X.shape[0])]
        clusters.put((X.shape[0], next(tiebreaker), X.copy(), np.array([index for index in range(X.shape[0])])))
        for i in range(self.max_iter):
            _, _, cluster, indices = clusters.get()

            labels = self.kmeans.fit_predict(cluster)

            labels0 = labels == 0
            cluster0 = cluster[labels0, :]
            indices0 = indices[labels0]

            labels1 = labels == 1
            cluster1 = cluster[labels1, :]
            indices1 = indices[labels1]

            clusters.put((cluster0.shape[0]*(-1), next(tiebreaker), cluster0, indices0))
            clusters.put((cluster1.shape[0]*(-1), next(tiebreaker), cluster1, indices1))

            for index, label in zip(indices, labels):
                output[index].append(label)

        return output

    def fit_predict(self, X):
        return self.fit(X).predict(X)
