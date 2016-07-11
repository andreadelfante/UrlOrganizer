from utils.UrlConverter import UrlConverter
from utils.UrlsEmbedding import UrlsEmbedding
import pandas as pd
import utils.Formatter as F

class RunExperiments:

    def __init__(self, direct, separator="\\t", scale="none"):
        file_url_codeUrl = direct + "seedsMap.txt"
        file_url_cluster = direct + "groundTruth.csv"
        file_embeddings_with_b = direct + "embeddings_with_b.txt"
        file_embeddings_no_b = direct + "embeddings_no_b.txt"

        converter = UrlConverter(file_url_cluster, file_url_codeUrl, separator)
        self.__embeddings_with_b = UrlsEmbedding(file_embeddings_with_b, scaling=scale)
        self.__embeddings_no_b = UrlsEmbedding(file_embeddings_no_b, scaling=scale)

        true_labels = converter.get_true_clusteringLabels
        cluster_size = len(set(true_labels))

        learned_labels_kmeans_with_b = self.__embeddings_with_b.clustering(type_clustering="kmeans",
                                                                           n_clusters=cluster_size)
        learned_labels_kmeans_no_b = self.__embeddings_no_b.clustering(type_clustering="kmeans",
                                                                       n_clusters=cluster_size)
        learned_labels_hdbscan_with_b = self.__embeddings_with_b.clustering(type_clustering="hdbscan")
        learned_labels_hdbscan_no_b = self.__embeddings_no_b.clustering(type_clustering="hdbscan")

        url_codes_with_b = self.__embeddings_with_b.get_urls
        url_codes_no_b = self.__embeddings_no_b.get_urls

        triple_list_kmeans_with_b = converter.get_triple_list(list_codes_url=url_codes_with_b,
                                                              learned_labels=learned_labels_kmeans_with_b)
        triple_list_kmeans_no_b = converter.get_triple_list(list_codes_url=url_codes_no_b,
                                                            learned_labels=learned_labels_kmeans_no_b)
        triple_list_hdbscan_with_b = converter.get_triple_list(list_codes_url=url_codes_with_b,
                                                               learned_labels=learned_labels_hdbscan_with_b)
        triple_list_hdbscan_no_b = converter.get_triple_list(list_codes_url=url_codes_no_b,
                                                             learned_labels=learned_labels_hdbscan_no_b)

        self.__metrics_kmeans_with_b = self.__embeddings_with_b.test_filter_urls(triple_list=triple_list_kmeans_with_b)
        self.__metrics_kmeans_no_b = self.__embeddings_no_b.test_filter_urls(triple_list=triple_list_kmeans_no_b)
        self.__metrics_hdbscan_with_b = self.__embeddings_with_b.test_filter_urls(triple_list=triple_list_hdbscan_with_b)
        self.__metrics_hdbscan_no_b = self.__embeddings_no_b.test_filter_urls(triple_list=triple_list_hdbscan_no_b)

    def plot_normalized_data_with_b(self, file_name="normalized_data_with_b"):
        return self.__embeddings_with_b.plot_normalized_data(file_name=file_name)

    def plot_normalized_data_no_b(self, file_name="normalized_data_no_b"):
        return self.__embeddings_no_b.plot_normalized_data(file_name=file_name)

    def get_dataframe_with_b(self):
        return self.__get_dataframe_metrics(metrics_kmeans=self.__metrics_kmeans_with_b,
                                            metrics_hdbscan=self.__metrics_hdbscan_with_b)

    def get_dataframe_no_b(self):
        return self.__get_dataframe_metrics(metrics_kmeans=self.__metrics_kmeans_no_b,
                                            metrics_hdbscan=self.__metrics_hdbscan_no_b)

    def __get_dataframe_metrics(self, metrics_kmeans, metrics_hdbscan):
        embedding_with_b = pd.DataFrame({
            "Homogeneity": [F.formatFloat(metrics_kmeans.get_homogeneity),
                            F.formatFloat(metrics_hdbscan.get_homogeneity)],
            "Completeness": [F.formatFloat(metrics_kmeans.get_completeness),
                             F.formatFloat(metrics_hdbscan.get_completeness)],
            "V-Measure": [F.formatFloat(metrics_kmeans.get_v_measure),
                          F.formatFloat(metrics_hdbscan.get_v_measure)],
            "Adj Rand index": [F.formatFloat(metrics_kmeans.get_adjuster_rand),
                               F.formatFloat(metrics_hdbscan.get_adjuster_rand)],
            "Adj Mutual info": [F.formatFloat(metrics_kmeans.get_mutual_information),
                                F.formatFloat(metrics_hdbscan.get_mutual_information)],
            "Silhouette": [F.formatFloat(metrics_kmeans.get_silhouette),
                           F.formatFloat(metrics_hdbscan.get_silhouette)]
        },
            index=["KMeans",
                   "HDBScan"]
        )

        return embedding_with_b