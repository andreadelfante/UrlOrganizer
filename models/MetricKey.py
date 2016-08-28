from enum import Enum


class MetricKey:

    class Clustering(Enum):
        KMeans = "kmeans"
        HDBScan = "hdbscan"

    def __init__(self, words, depth, window, clustering):
        self.__words = words
        self.__depth = depth
        self.__window = window
        self.__clustering = clustering.value

    def __hash__(self):
        result = hash(self.__words) + hash(self.__depth) + hash(self.__window)
        result += hash(self.__clustering)

        return result

    def __str__(self):
        return "MetricKey{" \
               "words=" + self.__words + "\n" \
               "depth=" + self.__depth + "\n" \
               "window=" + self.__window + "\n" \
               "clustering=" + self.__clustering + "\n" \
               "}"