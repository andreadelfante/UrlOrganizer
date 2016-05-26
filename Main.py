from utils.Data import Data
from utils.UrlMap import UrlMap
from utils.GroundTruth import GroundTruth

import pandas as pd

from hdbscan import HDBSCAN
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

ground_truth_path = "dataset/illinois/ground_truth/"
experimentation_path = "dataset/illinois/list_constraint/"

def main():
    url_map = UrlMap(file_path=experimentation_path + "urlMap.txt")
    data = Data(file_path=experimentation_path + "normalSkipgram.txt", url_map=url_map)

    embeddings = data.get_embeddings
    words = data.get_words

    tsne = TSNE(n_components=2)
    tsne.fit(embeddings)

    dbscan = DBSCAN(eps=0.8, min_samples=7)
    dbscan.fit(embeddings)
    dbscan_labels = dbscan.labels_

    hdbscan = HDBSCAN(min_cluster_size=10)
    hdbscan.fit(embeddings)
    hdbscan_labels = hdbscan.labels_

    kmeans = KMeans(n_clusters=30)
    kmeans.fit(embeddings)
    kmeans_labels = kmeans.labels_

    ground_truth = GroundTruth(file_name=ground_truth_path + "groundTruth.txt")
    ground_truth_clusters = ground_truth.get_clusters(words=words)

    metrics_df = pd.DataFrame([
        [
            # dbscan nocostraint
            metrics.homogeneity_score(ground_truth_clusters, dbscan_labels),
            metrics.completeness_score(ground_truth_clusters, dbscan_labels),
            metrics.v_measure_score(ground_truth_clusters, dbscan_labels),
            metrics.adjusted_rand_score(ground_truth_clusters, dbscan_labels),
            metrics.adjusted_mutual_info_score(ground_truth_clusters, dbscan_labels),
            metrics.silhouette_score(embeddings, dbscan_labels, metric='euclidean')
        ],
        [
            # hdbscan nocostraint
            metrics.homogeneity_score(ground_truth_clusters, hdbscan_labels),
            metrics.completeness_score(ground_truth_clusters, hdbscan_labels),
            metrics.v_measure_score(ground_truth_clusters, hdbscan_labels),
            metrics.adjusted_rand_score(ground_truth_clusters, hdbscan_labels),
            metrics.adjusted_mutual_info_score(ground_truth_clusters, hdbscan_labels),
            metrics.silhouette_score(embeddings, hdbscan_labels, metric='euclidean')
        ],
        [
            # kmeans nocostraint
            metrics.homogeneity_score(ground_truth_clusters, kmeans_labels),
            metrics.completeness_score(ground_truth_clusters, kmeans_labels),
            metrics.v_measure_score(ground_truth_clusters, kmeans_labels),
            metrics.adjusted_rand_score(ground_truth_clusters, kmeans_labels),
            metrics.adjusted_mutual_info_score(ground_truth_clusters, kmeans_labels),
            metrics.silhouette_score(embeddings, kmeans_labels, metric='euclidean')
        ]],
        index=[
            "DBSCAN",
            "HDBSCAN",
            "K-MEANS",
        ],
        columns=[
            "Homogeneity",
            "Completeness",
            "V-Measure core",
            "Adjusted Rand index",
            "Mutual Information",
            "Silhouette"
        ]
    )

    metrics_df.head() #show results

if __name__ == '__main__':
    main()