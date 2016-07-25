from enum import Enum

from models.MetricKey import MetricKey
from models.Metrics import Metrics
import plotly.plotly as py
import plotly.graph_objs as go


class Chart:

    __window_labeled = ["window: 2", "window: 3", "window: 5", "window: 7"]
    __window = ["2", "3", "5", "7"]

    __db_labeled = ["db: 100K", "db: 500K", "db: 1M"]
    __db = ["100000", "500000", "1000000"]

    __depth_labeled = ["depth: 10", "depth: 15", "depth: 20"]
    __depth = ["10", "15", "20"]

    class Metrics(Enum):
        v_measure = 0,
        ami = 1,
        ari = 2,
        silhouette = 3,
        homogeneity = 4,
        completeness = 5

    class Algorithm(Enum):
        left_skipgram = 0,
        normal_skipgram = 1

    def __init__(self, path, left_skipgram, normal_skipgram):
        self.__left_skipgram = self.__read_file(path=path + left_skipgram)
        self.__normal_skipgram = self.__read_file(path=path + normal_skipgram)

    def plot_linechart(self, metric, words_string, depth_string, clustering, mode='lines+markers'):
        x = self.__window

        arrays = self.__get_metrics_for_linechart(x, metric, words_string, depth_string, clustering)
        left_y = arrays[0]
        normal_y = arrays[1]

        trace0 = go.Scatter(
            x=self.__window_labeled,
            y=left_y,
            mode=mode,
            name=str(metric) + " left skipgram"
        )

        trace1 = go.Scatter(
            x=self.__window_labeled,
            y=normal_y,
            mode=mode,
            name=str(metric) + " normal skipgram"
        )

        data = [trace0, trace1]
        layout = dict(
            title="Metric: " + str(metric) + " - words: " + words_string + " - depth: " \
                  + depth_string + " - clustering: " + str(clustering)
        )

        fig = dict(data=data, layout=layout)
        return py.iplot(fig)

    def plot_heatmap_window_db(self, metric, algorithm, clustering, depth_string):
        y = self.__window
        x = self.__db

        y_labeled = self.__window_labeled
        x_labeled = self.__db_labeled

        z = []
        for el_y in y:
            el_z = []

            for el_x in x:
                key = MetricKey(el_x, depth_string, el_y, clustering)
                element = self.__get_value_heatmap(key, metric, algorithm)
                el_z.append(element)

            z.append(el_z)

        trace = go.Heatmap(
            x=x_labeled,
            y=y_labeled,
            z=z,
            showscale=False
        )

        fig = go.Figure(data=[trace])
        fig["layout"].update(
            title=str(metric) + " - " + str(algorithm) + " - " + str(clustering) + " - depth:" +
                  depth_string,
            annotations=self.__get_annotations_heatmap(x_labeled, y_labeled, z),
            autosize=True
        )

        return py.iplot(fig)

    def plot_heatmap_window_depth(self, metric, algorithm, clustering, db_string):
        y = self.__window
        x = self.__depth

        y_labeled = self.__window_labeled
        x_labeled = self.__depth_labeled

        z = []
        for el_y in y:
            el_z = []

            for el_x in x:
                key = MetricKey(db_string, el_x, el_y, clustering)
                element = self.__get_value_heatmap(key, metric, algorithm)
                el_z.append(element)

            z.append(el_z)

        trace = go.Heatmap(
            x=x_labeled,
            y=y_labeled,
            z=z,
            showscale=False
        )

        fig = go.Figure(data=[trace])
        fig["layout"].update(
            title=str(metric) + " - " + str(algorithm) + " - " + str(clustering) + " - db: " + db_string,
            annotations=self.__get_annotations_heatmap(x_labeled, y_labeled, z),
            width=700,
            height=700,
            autosize=True
        )

        return py.iplot(fig)

    def plot_heatmap_db_depth(self, metric, algorithm, clustering, window_string):
        y = self.__db
        x = self.__depth

        y_labeled = self.__db_labeled
        x_labeled = self.__depth_labeled

        z = []
        for el_y in y:
            el_z = []

            for el_x in x:
                key = MetricKey(el_y, el_x, window_string, clustering)
                element = self.__get_value_heatmap(key, metric, algorithm)
                el_z.append(element)

            z.append(el_z)

        trace = go.Heatmap(
            x=x_labeled,
            y=y_labeled,
            z=z,
            showscale=False
        )

        fig = go.Figure(data=[trace])
        fig["layout"].update(
            title=str(metric) + " - " + str(algorithm) + " - " + str(clustering) + " - window: " + window_string,
            annotations=self.__get_annotations_heatmap(x_labeled, y_labeled, z),
            width=700,
            height=700,
            autosize=True
        )

        return py.iplot(fig)

    def __get_annotations_heatmap(self, x, y, z):
        annotations = []
        for n, row in enumerate(z):
            for m, val in enumerate(row):
                annotations.append(
                    dict(
                        text=str(z[n][m]),
                        x=x[m], y=y[n],
                        xref='x1', yref='y1',
                        font=dict(color="white"),
                        showarrow=False
                    )
                )

        return annotations

    def __get_value_heatmap(self, key, metric, algorithm):
        elements = self.__get_metric_value(metric, key)

        if algorithm == Chart.Algorithm.left_skipgram:
            return elements[0]

        return elements[1]

    def __get_metrics_for_linechart(self, array, metric, words_string, depth_string, clustering_enum):
        array_left = []
        array_normal = []

        for element in array:
            key = MetricKey(words_string, depth_string, element, clustering_enum)
            elements = self.__get_metric_value(metric, key)
            element_left = elements[0]
            element_normal = elements[1]

            array_left.append(element_left)
            array_normal.append(element_normal)

        return array_left, array_normal

    def __get_metric_value(self, metric, key):
        left = self.__left_skipgram[key.__hash__()]
        normal = self.__normal_skipgram[key.__hash__()]

        element_left = None
        element_normal = None

        if metric == Chart.Metrics.v_measure:
            element_left = left.get_v_measure
            element_normal = normal.get_v_measure
        elif metric == Chart.Metrics.ami:
            element_left = left.get_mutual_information
            element_normal = normal.get_mutual_information
        elif metric == Chart.Metrics.ari:
            element_left = left.get_adjuster_rand
            element_normal = normal.get_adjuster_rand
        elif metric == Chart.Metrics.silhouette:
            element_left = left.get_silhouette
            element_normal = normal.get_silhouette
        elif metric == Chart.Metrics.homogeneity:
            element_left = left.get_homogeneity
            element_normal = normal.get_homogeneity
        elif metric == Chart.Metrics.completeness:
            element_left = left.get_completeness
            element_normal = normal.get_completeness

        return element_left, element_normal

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
                map[MetricKey(words, depth, window, MetricKey.Clustering.HDBScan).__hash__()] = self.__get_metrics(
                    array)

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

            if words is not None and depth is not None and window is not None:
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
