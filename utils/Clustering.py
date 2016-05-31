from time import time

import numpy as np
from hdbscan import HDBSCAN
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class Clustering:

    def __init__(self,
                 dbscan_eps=0.8,
                 dbscan_min_samples=7,
                 hdbscan_min_cluster_size=10,
                 kmeans_n_clusters=30
                 ):
        self.dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        self.hdbscan = HDBSCAN(min_cluster_size=hdbscan_min_cluster_size)
        self.kmeans = KMeans(n_clusters=kmeans_n_clusters)

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

    def plot_original_data(self,
                           embeddings,
                           tsne_n_components=2,
                           tsne_random_state=0
                           ):
        model = TSNE(n_components=tsne_n_components, random_state=tsne_random_state)
        np.set_printoptions(suppress=True)
        data = model.fit_transform(embeddings)
        plt.plot(data[:, 0], data[:, 1], 'o')
        output = "original_embeddings" + time() + ".png"
        plt.savefig(output)

    def plot_scaled_data(self,
                         normalized_embeddings
                         ):
        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        data = model.fit_transform(normalized_embeddings)
        plt.plot(data[:, 0], data[:, 1], 'o')
        output = "scaled_embeddings" + time() + ".png"
        plt.savefig(output)