import os.path
import numpy as np
from enum import Enum
from sklearn import preprocessing
from sklearn.cluster import KMeans
import hdbscan
from sklearn import metrics
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from time import time

_author_ = 'fabianalanotte'


class Scale(Enum):
    zscore = 1
    minmax = 2
    none = 3


class Clustering_algorithm(Enum):
    KMeans = 1
    HDBscan = 2


class LinksAnalyzer:
    def __init__(self, filename, scaling=Scale.zscore):
        assert isinstance(scaling, Scale), "the scaling parameter is not of type Scale"
        self.urls, self.embeddings = self.__read_embeddings(filename)
        self.normalized_embeddings = self.__scale(self.embeddings, scaling)

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
            print('scaling embeddings with minMaz normalization')
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

    def clustering(self, data, type_clustering=Clustering_algorithm.KMeans, n_clusters=10):
        assert isinstance(type_clustering,
                          Clustering_algorithm), "the input parameter is not of type Clustering_algorithm"
        if type_clustering == Clustering_algorithm.KMeans:
            print("Start running KMeans")
            estimator = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
            estimator.fit(data)
            return estimator.labels_

        print("Start running HDBscan")
        estimator = hdbscan.HDBSCAN(min_cluster_size=4)
        return estimator.fit_predict(self.normalized_embeddings)

    def test(self, real_labels, predicted_labels):
        assert len(real_labels) == len(predicted_labels), "Invalid input arguments"
        assert len(real_labels) > 0, "Invalid input arguments"
        assert isinstance(real_labels[0], int), "Type is not int"
        assert isinstance(predicted_labels[0], int), "Type is not int"
        print('homo compl   v-meas  ARI AMI')
        print('%.3f   %.3f   %.3f   %.3f   %.3f',
              metrics.homogeneity_score(real_labels, predicted_labels),
              metrics.completeness_score(real_labels, predicted_labels),
              metrics.v_measure_score(real_labels, predicted_labels),
              metrics.adjusted_rand_score(real_labels, predicted_labels),
              metrics.adjusted_mutual_info_score(real_labels, predicted_labels),
              metrics.silhouette_score(self.normalized_embeddings, predicted_labels,
                                       metric='euclidean'))



    # matching matrix
    def __get_confusion_table(self, ground_truth, predicted_labels):
        """
        :param ground_truth:
        :param predicted_labels:
        :return: confusion_table (numpy matrix)
        To print type:
            import pandas as pd
            pd.DataFrame(conf_table, index=set(ground_truth), columns=set(predicted_labels))
        """

        assert len(ground_truth) == len(predicted_labels), "Invalid input arguments"
        assert len(ground_truth) > 0, "Invalid input arguments"
        assert isinstance(ground_truth[0], int), "Type is not int"
        assert isinstance(predicted_labels[0], int), "Type is not int"

        # matrix(ground_truth x predicted_labels)
        conf_table = np.zeros((len(set(ground_truth)), len(set(predicted_labels))))
        real_clust = list(set(ground_truth))
        # it's needed because ground truth can have discontinuous cluster set
        clust_to_index = { real_clust[i]: i for i in range(len(real_clust)) }

        for real_clust in clust_to_index.values():
            for i in range(len(predicted_labels)):
                if clust_to_index[ground_truth[i]] == real_clust:
                    cluster_found = predicted_labels[i]
                    conf_table[real_clust, cluster_found] = conf_table[real_clust, cluster_found] + 1
        return conf_table


    def get_original_embedding(self):
        return self.embeddings

    def get_scaled_embeddings(self):
        return self.normalized_embeddings

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
