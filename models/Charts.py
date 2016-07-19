from enum import Enum

from models.MetricKey import MetricKey
from models.Metrics import Metrics
import plotly.plotly as py
import plotly.graph_objs as go


class Chart:

    class Metrics(Enum):
        v_measure = 0,
        ami = 1,
        ari = 2,
        silhouette = 3,
        homogeneity = 4,
        completeness = 5

    def __init__(self, path, left_skipgram, normal_skipgram):
        self.__left_skipgram = self.__read_file(path=path + left_skipgram)
        self.__normal_skipgram = self.__read_file(path=path + normal_skipgram)

    def plot_linechart(self, metric, words_string, depth_string, clustering, mode='lines+markers'):
        x = ['2', '3', '5', '7']

        arrays = self.__get_metrics_for_linechart(x, metric, words_string, depth_string, clustering)
        left_y = arrays[0]
        normal_y = arrays[1]

        trace0 = go.Scatter(
            x = x,
            y = left_y,
            mode = mode,
            name = str(metric) + " left skipgram"
        )

        trace1 = go.Scatter(
            x=x,
            y=normal_y,
            mode=mode,
            name=str(metric) + " normal skipgram"
        )

        data = [trace0, trace1]

        return py.iplot(data)

    def plot_heatchart(self):
        #TODO: complete it

    def __get_metrics_for_linechart(self, array, m, words_string, depth_string, clustering_enum):
        array_left = []
        array_normal = []

        for element in array:
            key = MetricKey(words_string, depth_string, element, clustering_enum)
            left = self.__left_skipgram[key.__hash__()]
            normal = self.__normal_skipgram[key.__hash__()]

            element_left = None
            element_normal = None

            if m == Chart.Metrics.v_measure:
                element_left = left.get_v_measure
                element_normal = normal.get_v_measure
            elif m == Chart.Metrics.ami:
                element_left = left.get_mutual_information
                element_normal = normal.get_mutual_information
            elif m == Chart.Metrics.ari:
                element_left = left.get_adjuster_rand
                element_normal = normal.get_adjuster_rand
            elif m == Chart.Metrics.silhouette:
                element_left = left.get_silhouette
                element_normal = normal.get_silhouette
            elif m == Chart.Metrics.homogeneity:
                element_left = left.get_homogeneity
                element_normal = normal.get_homogeneity
            elif m == Chart.Metrics.completeness:
                element_left = left.get_completeness
                element_normal = normal.get_completeness

            array_left.append(element_left)
            array_normal.append(element_normal)

        return array_left, array_normal

    def __read_file(self, path):
        in_file = open(path, "r")
        text = in_file.readlines()
        map = {}
        words = None
        depth = None
        window = None

        i = 0
        for line in text:
            if i % 5 == 0:
                array = line.split(sep=".")
                key = self.__get_key(array)
                words = key[0]
                depth = key[1]
                window = key[2]
            elif i % 5 == 2:
                array = line.split(sep=";")
                map[MetricKey(words, depth, window, MetricKey.Clustering.KMeans).__hash__()] = self.__get_metrics(array)
            elif i % 5 == 3:
                map[MetricKey(words, depth, window, MetricKey.Clustering.HDBScan).__hash__()] = self.__get_metrics(array)

            i += 1

        return map

    def __get_key(self, array):
        words = None
        depth = None
        window = None

        for element in array:
            if "words" in element:
                words = element.split(sep="words")
            if "depth" in element:
                depth = element.split(sep="depth")
            if "window" in element:
                window = element.split(sep="window")

            if words is not None and depth is not None and window is not None :
                return words[1], depth[1], window[1]

        return None

    def __get_metrics(self, array):
        i = 0
        ami = None
        ari = None
        completeness = None
        homogeneity = None
        silhouette = None
        vmeasure = None

        for element in array:
            if i == 1:
                ami = float(element)
            elif i == 2:
                ari = float(element)
            elif i == 3:
                completeness = float(element)
            elif i == 4:
                homogeneity = float(element)
            elif i == 5:
                silhouette = float(element)
            elif i == 6:
                vmeasure = float(element)

            if ami is not None and ari is not None and completeness is not None and homogeneity is not None \
                and silhouette is not None and vmeasure is not None:
                return Metrics(homogeneity=homogeneity, adjuster_rand=ari, mutual_information=ami,
                               completeness=completeness, silhouette=silhouette, v_measure=vmeasure)

            i += 1

        return None