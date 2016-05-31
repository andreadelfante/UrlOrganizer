import os.path
from enum import Enum
from time import time

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import sklearn.metrics as metrics

from utils.UrlConverter import UrlConverter


class Scale(Enum):
    zscore = 1
    minmax = 2
    none = 3


class Clustering_algorithm(Enum):
    KMeans = 1
    HDBscan = 2


class UrlsEmbedding:
    def __init__(self, file_path, scaling=Scale.zscore):
        assert isinstance(file_path, str), "file_path must be a string"
        self.urls, self.embeddings = self.__read_embeddings(file_path)
        self.normalized_embeddings = self.__scale(embeddings=self.embeddings, type_scale=scaling)

    def __read_embeddings(self, filename):
        assert os.path.isfile(filename), "the file %r does not exist" % filename
        in_file = open(filename, "r")
        text = in_file.readlines()
        urls = []
        matrix = []

        for line in text:
            tokens = line.rstrip().split(' ', 1)
            url = tokens[0]
            embedding = np.fromstring(tokens[1], dtype=float, sep=' ')
            matrix.append(embedding)
            urls.append(url)

        in_file.close()
        return urls, matrix

    def __scale(self, embeddings, type_scale):
        if type_scale == Scale.zscore:
            print('scaling embeddings with z score')
            return self.__scaling_minmax(embeddings)
        if type_scale == Scale.minmax:
            print('scaling embeddings with minMax normalization')
            return self.__scaling_zscore(embeddings)
        print('No scaling executed')
        return embeddings

    def __scaling_minmax(self, embeddings):
        min_max_scaler = preprocessing.MinMaxScaler()
        minmax_scale = min_max_scaler.fit_transform(embeddings)
        return minmax_scale

    def __scaling_zscore(self, embeddings):
        standard_scale = preprocessing.scale(embeddings)
        return standard_scale

    def clustering(self, type_clustering=Clustering_algorithm.KMeans, n_clusters=10):
        assert isinstance(type_clustering,
                          Clustering_algorithm), "the input parameter is not of type Clustering_algorithm"
        if type_clustering == Clustering_algorithm.KMeans:
            print("Start running KMeans")
            estimator = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
            estimator.fit(self.normalized_embeddings)
            return estimator.labels_

        print("Start running HDBscan")
        estimator = hdbscan.HDBSCAN(min_cluster_size=4)
        return estimator.fit_predict(self.normalized_embeddings)

    def test(self, url_converter, learned_labels, metric='euclidean'):
        assert isinstance(url_converter, UrlConverter), "url_converter must be an UrlConverter object"
        real_labels = url_converter.get_ordered_labels(listUrl=self.urls)

        homogeneity = metrics.homogeneity_score(real_labels, learned_labels)
        completness = metrics.completeness_score(real_labels, learned_labels)
        v_measure = metrics.v_measure_score(real_labels, learned_labels)
        adjuster_rand = metrics.adjusted_rand_score(real_labels, learned_labels)
        mutual_information = metrics.adjusted_mutual_info_score(real_labels, learned_labels)
        silhouette = metrics.silhouette_score(self.normalized_embeddings, learned_labels, metric=metric)

        print("Homogeneity: " + str(homogeneity))
        print("Completeness: " + str(completness))
        print("V-Measure core: " + str(v_measure))
        print("Adjusted Rand index: " + str(adjuster_rand))
        print("Mutual Information: " + str(mutual_information))
        print("Silhouette: " + str(silhouette))

    @property
    def get_original_embedding(self):
        return self.embeddings

    @property
    def get_scaled_embeddings(self):
        return self.normalized_embeddings

    @property
    def get_words(self):
        return self.urls

    def plot_original_data(self):
        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        data = model.fit_transform(self.embeddings)
        plt.plot(data[:, 0], data[:, 1], 'o')
        output = "original_embeddings" + time() + ".png"
        plt.savefig(output)

    def plot_scaled_data(self):
        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        data = model.fit_transform(self.normalized_embeddings)
        plt.plot(data[:, 0], data[:, 1], 'o')
        output = "scaled_embeddings" + time() + ".png"
        plt.savefig(output)
