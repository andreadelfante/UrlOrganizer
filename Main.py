from utils.Metrics import Metrics
from utils.Clustering import Clustering
from utils.Data import Data
from utils.UrlMap import UrlMap
from utils.GroundTruth import GroundTruth


embeddings_file = "leftSkipgramWithB.txt"

ground_truth_path = "dataset/illinois/ground_truth/"
experimentation_path = "dataset/illinois/"

list_constraint = "list_constraint/"
no_constraint = "no_constraint/"


def main():
    lc_url_map = UrlMap(file_path=experimentation_path + list_constraint + "urlMap.txt")
    nc_url_map = UrlMap(file_path=experimentation_path + no_constraint + "urlMap.txt")

    lc_data = Data(file_path=experimentation_path + list_constraint + embeddings_file, url_map=lc_url_map)
    nc_data = Data(file_path=experimentation_path + no_constraint + embeddings_file, url_map=nc_url_map)

    lc_embeddings = lc_data.get_embeddings
    nc_embeddings = nc_data.get_embeddings

    lc_clustering = Clustering()
    lc_clustering.fit(lc_embeddings)

    nc_clustering = Clustering()
    nc_clustering.fit(nc_embeddings)

    ground_truth = GroundTruth(file_name=ground_truth_path + "groundTruth.txt")
    ground_truth_lc = ground_truth.get_clusters(words=lc_data.get_words)
    ground_truth_nc = ground_truth.get_clusters(words=nc_data.get_words)

    metrics = Metrics(columns=[
        "Homogeneity",
        "Completeness",
        "V-Measure core",
        "Adjusted Rand index",
        "Mutual Information",
        "Silhouette"
    ])

    metrics.addRow(index="NoCostraint - DBSCAN", ground_truth=ground_truth_nc, labels=nc_clustering.get_dbscan_labels, embeddings=nc_embeddings)
    metrics.addRow(index="NoCostraint - HDBSCAN", ground_truth=ground_truth_nc, labels=nc_clustering.get_hdbscan_labels, embeddings=nc_embeddings)
    metrics.addRow(index="NoCostraint - K-MEANS", ground_truth=ground_truth_nc, labels=nc_clustering.get_kmeans_labels, embeddings=nc_embeddings)

    metrics.addRow(index="ListCostraint - DBSCAN", ground_truth=ground_truth_lc, labels=lc_clustering.get_dbscan_labels, embeddings=lc_embeddings)
    metrics.addRow(index="ListCostraint - HDBSCAN", ground_truth=ground_truth_lc, labels=lc_clustering.get_hdbscan_labels, embeddings=lc_embeddings)
    metrics.addRow(index="ListCostraint - K-MEANS", ground_truth=ground_truth_lc, labels=lc_clustering.get_kmeans_labels, embeddings=lc_embeddings)


    metrics.show()

if __name__ == '__main__':
    main()