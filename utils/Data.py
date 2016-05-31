import os
from enum import Enum

import numpy as np
from sklearn import preprocessing

from utils.UrlMap import UrlMap

class Scale(Enum):
    zscore = 1
    minmax = 2
    none = 3

class UrlsEmbedding:

    def __init__(self, file_path, url_map=None, scaling=Scale.zscore):
        assert isinstance(file_path, str), "file_path must be a string"
        assert url_map is None or isinstance(url_map, UrlMap), "url_map must be an UrlMap object or None"
        self.urls, self.embeddings = self.__read_embeddings(file_path)
        self.normalized_embeddings = self.scale(self.embeddings, scaling)

        if isinstance(url_map, UrlMap):
            self.urls = [url_map[id_url] for id_url in self.urls]

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

    def scale(self, embeddings, type_scale):
        if type_scale == Scale.zscore:
            print('scaling embeddings with z score')
            return self.scaling_minmax(embeddings)
        if type_scale == Scale.minmax:
            print('scaling embeddings with minMaz normalization')
            return self.scaling_zscore(embeddings)
        print('No scaling executed')
        return embeddings

    def scaling_minmax(self, embeddings):
        min_max_scaler = preprocessing.MinMaxScaler()
        minmax_scale = min_max_scaler.fit_transform(embeddings)
        return minmax_scale

    def scaling_zscore(self, embeddings):
        standard_scale = preprocessing.scale(embeddings)
        return standard_scale

    @property
    def get_embeddings(self):
        return self.embeddings

    @property
    def get_normalized_embeddings(self):
        return self.normalized_embeddings

