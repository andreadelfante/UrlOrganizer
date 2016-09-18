import pandas as pd

from models.Metrics import Metrics
from models.UrlsEmbedding import Clustering_algorithm


def formatFloat(value):
    '''
    Format a float value
    :param value: a float value
    :return: a float value formatted
    '''

    return "{0:.2f}".format(value)


def get_dataframe_metrics(metrics_kmeans, metrics_hdbscan):
    assert isinstance(metrics_kmeans, Metrics), "metrics_kmeans must be a Metrics object"
    assert isinstance(metrics_hdbscan, Metrics), "metrics_hdbscan must be a Metrics object"

    result = pd.DataFrame({
        "Homogeneity": [formatFloat(metrics_kmeans.get_homogeneity),
                        formatFloat(metrics_hdbscan.get_homogeneity)],
        "Completeness": [formatFloat(metrics_kmeans.get_completeness),
                         formatFloat(metrics_hdbscan.get_completeness)],
        "V-Measure": [formatFloat(metrics_kmeans.get_v_measure),
                      formatFloat(metrics_hdbscan.get_v_measure)],
        "Adj Rand index": [formatFloat(metrics_kmeans.get_adjuster_rand),
                           formatFloat(metrics_hdbscan.get_adjuster_rand)],
        "Adj Mutual info": [formatFloat(metrics_kmeans.get_mutual_information),
                            formatFloat(metrics_hdbscan.get_mutual_information)],
        "Silhouette": [formatFloat(metrics_kmeans.get_silhouette),
                       formatFloat(metrics_hdbscan.get_silhouette)]
    },
        index=["KMeans",
               "HDBScan"]
    )

    return result

def get_dataframe_metrics_just_one(metrics, clustering_algorithm):
    assert isinstance(metrics, Metrics), "metrics must be a Metrics object"
    assert isinstance(clustering_algorithm, Clustering_algorithm) or isinstance(clustering_algorithm, str), \
        "clustering_algorithm must be a string or Clustering_algorithm enum"

    index = []
    if clustering_algorithm == Clustering_algorithm.KMeans or clustering_algorithm == Clustering_algorithm.KMeans.value:
        index.append("KMeans")
    elif clustering_algorithm == Clustering_algorithm.HDBscan or clustering_algorithm == Clustering_algorithm.HDBscan.value:
        index.append("HDBScan")
    else:
        raise RuntimeError("No valid clustering algorithm in input")

    result = pd.DataFrame({
        "Homogeneity": [formatFloat(metrics.get_homogeneity)],
        "Completeness": [formatFloat(metrics.get_completeness)],
        "V-Measure": [formatFloat(metrics.get_v_measure)],
        "Adj Rand index": [formatFloat(metrics.get_adjuster_rand)],
        "Adj Mutual info": [formatFloat(metrics.get_mutual_information)],
        "Silhouette": [formatFloat(metrics.get_silhouette)]
    },
        index=index
    )

    return result