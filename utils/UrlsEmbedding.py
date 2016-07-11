import os.path
from enum import Enum
from time import time

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.preprocessing import Normalizer
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

from models.Metrics import Metrics


class Scale(Enum):
    zscore = "zscore"
    minmax = "minmax"
    none = "none"
    l2 = "l2"


class Clustering_algorithm(Enum):
    KMeans = "kmeans"
    HDBscan = "hdbscan"


class UrlsEmbedding:

    def __init__(self, file_path, scaling=Scale.none):
        assert isinstance(file_path, str), "file_path must be a string"
        self.__urls, self.__embeddings = self.__read_embeddings(file_path)
        self.__normalized_embeddings = self.__scale(embeddings=self.__embeddings, type_scale=scaling)

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
        return np.array(urls), np.array(matrix)

    def __scale(self, embeddings, type_scale):
        if len(embeddings) == 0:
            return np.array([])

        if type_scale == Scale.zscore.value:
            print('scaling embeddings with z score')
            return self.__scaling_zscore(embeddings)
        if type_scale == Scale.minmax.value:
            print('scaling embeddings with minMax normalization')
            return self.__scaling_minmax(embeddings)
        if type_scale == Scale.l2.value:
            return self.__scaling_l2(embeddings)

        print('No scaling executed')
        return embeddings

    def __scaling_minmax(self, embeddings):
        min_max_scaler = preprocessing.MinMaxScaler()
        minmax_scale = min_max_scaler.fit_transform(embeddings)
        return minmax_scale

    def __scaling_zscore(self, embeddings):
        standard_scale = preprocessing.scale(embeddings)
        return standard_scale

    def __scaling_l2(self, embeddings, copy=False):
        normalizer = Normalizer(copy=copy)
        return normalizer.fit_transform(embeddings)

    def clustering(self, type_clustering=Clustering_algorithm.KMeans, n_clusters=10):
        if type_clustering is Clustering_algorithm:
            type_clustering = type_clustering.value

        if type_clustering == Clustering_algorithm.KMeans.value:
            print("Start running KMeans")
            estimator = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
            estimator.fit(self.__normalized_embeddings)
            return estimator.labels_

        if type_clustering == Clustering_algorithm.HDBscan.value:
            print("Start running HDBscan")
            estimator = hdbscan.HDBSCAN(min_cluster_size=5)
            return estimator.fit_predict(self.__normalized_embeddings)

        print("No selected clustering method")
        return None

    def test(self, real_labels, learned_labels, metric='euclidean'):
        assert isinstance(real_labels, list) or isinstance(real_labels, np.ndarray), "real_labels must be a list or a numpy array"
        assert isinstance(learned_labels, list) or isinstance(learned_labels, np.ndarray), "learned_labels must be a list or a numpy array"
        assert len(real_labels) == len(learned_labels), "real_labels and learned_labels must have the same length"

        homogeneity = metrics.homogeneity_score(real_labels, learned_labels)
        completness = metrics.completeness_score(real_labels, learned_labels)
        v_measure = metrics.v_measure_score(real_labels, learned_labels)
        adjuster_rand = metrics.adjusted_rand_score(real_labels, learned_labels)
        mutual_information = metrics.adjusted_mutual_info_score(real_labels, learned_labels)
        silhouette = metrics.silhouette_score(self.__normalized_embeddings, learned_labels, metric=metric)

        print("Homogeneity: " + str(homogeneity))
        print("Completeness: " + str(completness))
        print("V-Measure core: " + str(v_measure))
        print("Adjusted Rand index: " + str(adjuster_rand))
        print("Mutual Information: " + str(mutual_information))
        print("Silhouette: " + str(silhouette))

        return Metrics(homogeneity, completness, v_measure, adjuster_rand, mutual_information, silhouette)

    def test_filter_urls(self, triple_list):
        assert isinstance(triple_list, list) or isinstance(triple_list, np.ndarray), "triple_list must be a list or a numpy array"

        filtered_triple_list = [element for element in triple_list if element[1] != -1]
        real_labels = []
        learned_labels = []

        for triple in filtered_triple_list:
            real_labels.append(triple[1])
            learned_labels.append(triple[2])

        real_labels = np.array(real_labels)
        learned_labels = np.array(learned_labels)

        return self.test(real_labels=real_labels,learned_labels=learned_labels)

    @property
    def get_original_embedding(self):
        return self.__embeddings

    @property
    def get_scaled_embeddings(self):
        return self.__normalized_embeddings

    @property
    def get_urls(self):
        return self.__urls

    def plot_original_data(self, file_name="original_embeddings" + str(time())):
        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        data = model.fit_transform(self.__embeddings)
        plt.plot(data[:, 0], data[:, 1], 'o')
        output = file_name + ".png"
        plt.savefig(output)

        return plt

    def plot_normalized_data(self, file_name="scaled_embeddings" + str(time())):
        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        data = model.fit_transform(self.__normalized_embeddings)
        plt.plot(data[:, 0], data[:, 1], 'o')
        output = file_name + ".png"
        plt.savefig(output)

        return plt