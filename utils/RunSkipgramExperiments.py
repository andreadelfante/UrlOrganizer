from models.UrlsEmbedding import UrlsEmbedding

import utils.Formatter as F
from models.UrlConverter import UrlConverter


class RunSkipgramExperiments:

    def __init__(self, direct, separator="\\t", scale="none", intersect=False):
        file_url_codeUrl = direct + "seedsMap.txt"
        file_url_cluster = direct + "groundTruth.csv"
        file_embeddings_with_b = direct + "embeddings_with_b.txt"
        file_embeddings_no_b = direct + "embeddings_no_b.txt"
        file_embeddings_normal = direct + "embeddings_normal.txt"

        converter = UrlConverter(file_url_cluster, file_url_codeUrl, separator)
        self.__embeddings_with_b = UrlsEmbedding(file_embeddings_with_b, scaling=scale)
        self.__embeddings_no_b = UrlsEmbedding(file_embeddings_no_b, scaling=scale)
        self.__embeddings_normal = UrlsEmbedding(file_embeddings_normal, scaling=scale)

        if intersect:
            self.__embeddings_normal.intersect(self.__embeddings_with_b.get_urls)
            self.__embeddings_with_b.intersect(self.__embeddings_normal.get_urls)
            self.__embeddings_no_b.intersect(self.__embeddings_normal.get_urls)

        true_labels = converter.get_true_clusteringLabels
        cluster_size = len(set(true_labels))

        learned_labels_kmeans_with_b = self.__embeddings_with_b.clustering(type_clustering="kmeans",
                                                                           n_clusters=cluster_size)
        learned_labels_kmeans_no_b = self.__embeddings_no_b.clustering(type_clustering="kmeans",
                                                                       n_clusters=cluster_size)
        learned_labels_kmeans_normal = self.__embeddings_normal.clustering(type_clustering="kmeans",
                                                                           n_clusters=cluster_size)
        learned_labels_hdbscan_with_b = self.__embeddings_with_b.clustering(type_clustering="hdbscan")
        learned_labels_hdbscan_no_b = self.__embeddings_no_b.clustering(type_clustering="hdbscan")
        learned_labels_hdbscan_normal = self.__embeddings_normal.clustering(type_clustering="hdbscan")

        url_codes_with_b = self.__embeddings_with_b.get_urls
        url_codes_no_b = self.__embeddings_no_b.get_urls
        url_codes_normal = self.__embeddings_normal.get_urls

        triple_list_kmeans_with_b = converter.get_triple_list(list_codes_url=url_codes_with_b,
                                                              learned_labels=learned_labels_kmeans_with_b)
        triple_list_kmeans_no_b = converter.get_triple_list(list_codes_url=url_codes_no_b,
                                                            learned_labels=learned_labels_kmeans_no_b)
        triple_list_kmeans_normal = converter.get_triple_list(list_codes_url=url_codes_normal,
                                                              learned_labels=learned_labels_kmeans_normal)
        triple_list_hdbscan_with_b = converter.get_triple_list(list_codes_url=url_codes_with_b,
                                                               learned_labels=learned_labels_hdbscan_with_b)
        triple_list_hdbscan_no_b = converter.get_triple_list(list_codes_url=url_codes_no_b,
                                                             learned_labels=learned_labels_hdbscan_no_b)
        triple_list_hdbscan_normal = converter.get_triple_list(list_codes_url=url_codes_normal,
                                                               learned_labels=learned_labels_hdbscan_normal)

        self.__metrics_kmeans_with_b = self.__embeddings_with_b.test_filter_urls(triple_list=triple_list_kmeans_with_b)
        self.__metrics_kmeans_no_b = self.__embeddings_no_b.test_filter_urls(triple_list=triple_list_kmeans_no_b)
        self.__metrics_kmeans_normal = self.__embeddings_normal.test_filter_urls(triple_list=triple_list_kmeans_normal)
        self.__metrics_hdbscan_with_b = self.__embeddings_with_b.test_filter_urls(triple_list=triple_list_hdbscan_with_b)
        self.__metrics_hdbscan_no_b = self.__embeddings_no_b.test_filter_urls(triple_list=triple_list_hdbscan_no_b)
        self.__metrics_hdbscan_normal = self.__embeddings_normal.test_filter_urls(triple_list=triple_list_hdbscan_normal)

    def plot_normalized_data_with_b(self, file_name="normalized_data_with_b"):
        return self.__embeddings_with_b.plot_normalized_data(file_name=file_name)

    def plot_normalized_data_no_b(self, file_name="normalized_data_no_b"):
        return self.__embeddings_no_b.plot_normalized_data(file_name=file_name)

    def plot_normalized_data_normal(self, file_name="normalized_data_normal"):
        return self.__embeddings_normal.plot_normalized_data(file_name=file_name)

    def get_dataframe_with_b(self):
        return F.get_dataframe_metrics(metrics_kmeans=self.__metrics_kmeans_with_b,
                                            metrics_hdbscan=self.__metrics_hdbscan_with_b)

    def get_dataframe_no_b(self):
        return F.get_dataframe_metrics(metrics_kmeans=self.__metrics_kmeans_no_b,
                                            metrics_hdbscan=self.__metrics_hdbscan_no_b)

    def get_dataframe_normal(self):
        return F.get_dataframe_metrics(metrics_kmeans=self.__metrics_kmeans_normal,
                                            metrics_hdbscan=self.__metrics_hdbscan_normal)