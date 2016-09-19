import os.path
from enum import Enum
from time import time

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
import utils.TfIdf as tfidf
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

    def __init__(self, urls, embeddings, scaling=Scale.none):
        '''
        Create new instance of UrlsEmbedding with url and embedding array
        :param urls: a numpy array of urls
        :param embeddings: a numpy array of embeddings
        '''
        assert isinstance(urls, np.ndarray), "urls must be a numpy array"
        assert isinstance(embeddings, np.ndarray), "embeddings must be a numpy array"
        assert urls.shape[0] == embeddings.shape[0], "urls and embeddings size must be the same"

        self.__urls = urls
        self.__embeddings = embeddings
        self.__normalized_embeddings = self.__scale(self.__embeddings, scaling)

    @classmethod
    def init_from_vertex(cls, file_path, dimension_reduction, scaling=Scale.none):
        '''
        Create a new instance of UrlsEmbedding from vertex.txt and fit text with tfidf
        :param file_path: a string containing the file path of vertex.txt
        :param scaling: an enum (class Scale) for scaling.
        '''
        assert isinstance(file_path, str), "file_path must be a string"
        urls, embeddings = tfidf.read_vertex_file(file_path)
        matrix = tfidf.fit(embeddings, dimension_reduction)
        return UrlsEmbedding(urls, matrix, scaling)

    @classmethod
    def init_from_embeddings(cls, file_path, scaling=Scale.none):
        '''
        Create a new instance of UrlsEmbedding from embeddings file txt
        :param file_path: a string containing the file path of embeddings file txt
        :param scaling: an enum (class Scale) for scaling.
        '''
        assert isinstance(file_path, str), "file_path must be a string"
        urls, embeddings = UrlsEmbedding.__read_embeddings(file_path)
        return UrlsEmbedding(urls, embeddings, scaling)

    @classmethod
    def __read_embeddings(cls, filename):
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

    def intersect(self, urls):
        '''
        Perform an intersection with urls in place
        :param urls: an nparray
        '''
        assert isinstance(urls, np.ndarray) or isinstance(urls, list), "urls must be an np array or list"

        print("Original urls: " + str(len(self.__urls)))

        if isinstance(urls, np.ndarray):
            urls = urls.tolist()

        self.__urls = self.__urls.tolist()
        self.__embeddings = self.__embeddings.tolist()
        self.__normalized_embeddings = self.__normalized_embeddings.tolist()

        i = len(self.__urls) - 1
        while i >= 0:
            if not self.__urls[i] in urls:
                del self.__urls[i]
                del self.__embeddings[i]
                del self.__normalized_embeddings[i]

            i -= 1

        self.__urls = np.array(self.__urls)
        self.__embeddings = np.array(self.__embeddings)
        self.__normalized_embeddings = np.array(self.__normalized_embeddings)

        print("Intersected urls: " + str(len(self.__urls)))

    def concatenate(self, another_embedding):
        '''
        Perform a concatenation from this and another embedding
        :param another_embedding: another UrlsEmbedding object
        :return:
        '''
        assert isinstance(another_embedding, UrlsEmbedding), "another_embedding must be an UrlsEmbedding object"

        concatenate_embeddings = []

        for i in range(self.__urls.size):
            url = self.__urls[i]

            found = False
            j = 0
            while j in range(another_embedding.__urls.size) and not found:
                another_url = another_embedding.__urls[j]

                if url == another_url:
                    concatenate_embeddings.append(
                        np.concatenate((self.__normalized_embeddings[i], another_embedding.__normalized_embeddings[j]))
                    )
                    found = True
                else:
                    j += 1

            if not found:
                raise RuntimeError(str(url) + " not found embedding in another_embedding to concatenate")

        self.__normalized_embeddings = np.array(concatenate_embeddings)

    def clustering(self, type_clustering=Clustering_algorithm.KMeans, n_clusters=10):
        '''
        Perform clustering
        :param type_clustering: an enum (class Clustering_algorithm) to perform cluster operation
        :param n_clusters: clusters number
        :return: clustering labels (array)
        '''

        if type_clustering == Clustering_algorithm.KMeans.value or type_clustering == Clustering_algorithm.KMeans:
            print("Start running KMeans")
            estimator = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
            estimator.fit(self.__normalized_embeddings)
            return estimator.labels_

        if type_clustering == Clustering_algorithm.HDBscan.value or type_clustering == Clustering_algorithm.HDBscan:
            print("Start running HDBscan")
            estimator = hdbscan.HDBSCAN(min_cluster_size=5)
            return estimator.fit_predict(self.__normalized_embeddings)

        print("No selected clustering method")
        return None

    def test(self, real_labels, learned_labels, metric='euclidean'):
        '''
        Perform test to meeasure my metrics
        :param real_labels: labels from groundtruth (np array)
        :param learned_labels: labels from clustering (np array)
        :param metric: a metric (default: euclidean)
        :return: a Metrics object containing the results
        '''

        assert isinstance(real_labels, list) or isinstance(real_labels,
                                                           np.ndarray), "real_labels must be a list or a numpy array"
        assert isinstance(learned_labels, list) or isinstance(learned_labels,
                                                              np.ndarray), "learned_labels must be a list or a numpy array"
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
        '''
        Perform test to measure my metrics using a triple list [(code_url, groundtruth_label, clustering_label)]. this triple list is a np array
        :param triple_list: an np array of triple
        :return: a Metrics object containing the results
        '''

        assert isinstance(triple_list, list) or isinstance(triple_list,
                                                           np.ndarray), "triple_list must be a list or a numpy array"

        filtered_triple_list = [element for element in triple_list if element[1] != -1 and element[1] != '-1']

        real_labels = []
        learned_labels = []

        for triple in filtered_triple_list:
            real_labels.append(triple[1])
            learned_labels.append(triple[2])

        real_labels = np.array(real_labels)
        learned_labels = np.array(learned_labels)

        return self.test(real_labels=real_labels, learned_labels=learned_labels)

    @property
    def get_original_embedding(self):
        '''
        :return: returns embeddings without normalization
        '''

        return self.__embeddings

    @property
    def get_scaled_embeddings(self):
        '''
        :return: returns embeddings with normalization
        '''

        return self.__normalized_embeddings

    @property
    def get_urls(self):
        '''
        :return: returns urls
        '''

        return self.__urls

    def plot_original_data(self, file_name="original_embeddings" + str(time())):
        '''
        Plot original data in a image
        :param file_name: file name
        :return: return plt object to print images on notebooks
        '''

        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        data = model.fit_transform(self.__embeddings)
        plt.plot(data[:, 0], data[:, 1], 'o')
        output = file_name + ".png"
        plt.savefig(output)

        return plt

    def plot_normalized_data(self, file_name="scaled_embeddings" + str(time())):
        '''
        Plot normalized data in a image
        :param file_name: file name
        :return: return plt object to primt images on notebooks
        '''
        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        data = model.fit_transform(self.__normalized_embeddings)
        plt.plot(data[:, 0], data[:, 1], 'o')
        output = file_name + ".png"
        plt.savefig(output)

        return plt

    def __scale(self, embeddings, type_scale):
        if len(embeddings) == 0:
            return np.array([])

        if type_scale == Scale.zscore.value or type_scale == Scale.zscore:
            print('scaling embeddings with z score')
            return self.__scaling_zscore(embeddings)
        if type_scale == Scale.minmax.value or type_scale == Scale.minmax:
            print('scaling embeddings with minMax normalization')
            return self.__scaling_minmax(embeddings)
        if type_scale == Scale.l2.value or type_scale == Scale.l2:
            print('scaling embeddings with L2 normalization')
            return self.__scaling_l2(embeddings)

        print('No scaling executed')
        return embeddings

    def __scaling_minmax(self, embeddings):
        min_max_scaler = preprocessing.MinMaxScaler(copy=True)
        minmax_scale = min_max_scaler.fit_transform(embeddings)
        return minmax_scale

    def __scaling_zscore(self, embeddings):
        standard_scale = preprocessing.scale(embeddings, copy=True)
        return standard_scale

    def __scaling_l2(self, embeddings):
        normalizer = Normalizer(copy=True)
        return normalizer.fit_transform(embeddings)
