import pandas as pd

from models.Metrics import Metrics


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