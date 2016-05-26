from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

class Clustering:

    def __init__(self):
        self.dbscan = DBSCAN(eps=0.8, min_samples=7)
        self.hdbscan = HDBSCAN(min_cluster_size=10)
        self.kmeans = KMeans(n_clusters=30)

    def fit(self, embeddings):
        self.dbscan.fit(embeddings)
        self.hdbscan.fit(embeddings)
        self.kmeans.fit(embeddings)

    @property
    def get_dbscan_labels(self):
        return self.dbscan.labels_

    @property
    def get_hdbscan_labels(self):
        return self.hdbscan.labels_

    @property
    def get_kmeans_labels(self):
        return self.kmeans.labels_