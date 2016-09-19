from models.UrlsEmbedding import UrlsEmbedding

import utils.Formatter as F
from models.UrlConverter import UrlConverter


class RunLineExperiments:

    def __init__(self, direct, prefix, db_normal_skipgram, window_normal_skipgram, depth_normal_skipgram, iteractions,
                 separator="\\t", scale="none", intersect=False):
        self.__config_normal_skipgram = "words" + db_normal_skipgram + \
                                 ".depth" + depth_normal_skipgram + \
                                 ".window" + window_normal_skipgram + \
                                 ".iteractions" + iteractions

        direct_normal_skipgram = direct + prefix + self.__config_normal_skipgram + "/"

        file_url_codeUrl = direct_normal_skipgram + "seedsMap.txt"
        file_url_cluster = direct_normal_skipgram + "groundTruth.csv"
        file_embeddings_normal_skipgram = direct_normal_skipgram + "embeddings_normal.txt"
        file_embeddings_line_first = direct + "embeddings_line_first.txt"
        file_embeddings_line_second = direct + "embeddings_line_second.txt"

        converter = UrlConverter(file_url_cluster, file_url_codeUrl, separator)
        self.__embeddings_normal_skipgram = UrlsEmbedding.init_from_embeddings(file_embeddings_normal_skipgram, scale)
        self.__embeddings_line_first = UrlsEmbedding.init_from_embeddings(file_embeddings_line_first, scale)
        self.__embeddings_line_second = UrlsEmbedding.init_from_embeddings(file_embeddings_line_second, scale)

        if intersect:
            self.__embeddings_line_first.intersect(self.__embeddings_normal_skipgram.get_urls)
            self.__embeddings_line_second.intersect(self.__embeddings_normal_skipgram.get_urls)
            self.__embeddings_normal_skipgram.intersect(self.__embeddings_line_first.get_urls)

        true_labels = converter.get_true_clusteringLabels
        cluster_size = len(set(true_labels))

        learned_labels_kmeans_normal_skipgram = self.__embeddings_normal_skipgram.clustering(type_clustering="kmeans",
                                                                                             n_clusters=cluster_size)
        learned_labels_kmeans_line_first = self.__embeddings_line_first.clustering(type_clustering="kmeans",
                                                                             n_clusters=cluster_size)
        learned_labels_kmeans_line_second = self.__embeddings_line_second.clustering(type_clustering="kmeans",
                                                                                     n_clusters=cluster_size)

        learned_labels_hdbscan_normal_skipgram = self.__embeddings_normal_skipgram.clustering(type_clustering="hdbscan")
        learned_labels_hdbscan_line_first = self.__embeddings_line_first.clustering(type_clustering="hdbscan")
        learned_labels_hdbscan_line_second = self.__embeddings_line_second.clustering(type_clustering="hdbscan")

        url_codes_normal_skipgram = self.__embeddings_normal_skipgram.get_urls
        url_codes_line_first = self.__embeddings_line_first.get_urls
        url_codes_line_second = self.__embeddings_line_second.get_urls

        triple_list_kmeans_normal_skipgram = converter.get_triple_list(list_codes_url=url_codes_normal_skipgram,
                                                                       learned_labels=
                                                                       learned_labels_kmeans_normal_skipgram)
        triple_list_kmeans_line_first = converter.get_triple_list(list_codes_url=url_codes_line_first,
                                                            learned_labels=learned_labels_kmeans_line_first)
        triple_list_kmeans_line_second = converter.get_triple_list(list_codes_url=url_codes_line_second,
                                                                   learned_labels=learned_labels_kmeans_line_second)

        triple_list_hdbscan_normal_skipgram = converter.get_triple_list(list_codes_url=url_codes_normal_skipgram,
                                                                        learned_labels=
                                                                        learned_labels_hdbscan_normal_skipgram)
        triple_list_hdbscan_line_first = converter.get_triple_list(list_codes_url=url_codes_line_first,
                                                             learned_labels=learned_labels_hdbscan_line_first)
        triple_list_hdbscan_line_second = converter.get_triple_list(list_codes_url=url_codes_line_second,
                                                                    learned_labels=learned_labels_hdbscan_line_second)

        self.__metrics_kmeans_normal_skipgram = self.__embeddings_normal_skipgram.test_filter_urls(
            triple_list_kmeans_normal_skipgram)
        self.__metrics_kmeans_line_first = self.__embeddings_line_first.test_filter_urls(triple_list_kmeans_line_first)
        self.__metrics_kmeans_line_second = self.__embeddings_line_second.test_filter_urls(triple_list_kmeans_line_second)

        self.__metrics_hdbscan_normal_skipgram = self.__embeddings_normal_skipgram.test_filter_urls(
            triple_list_hdbscan_normal_skipgram)
        self.__metrics_hdbscan_line_first = self.__embeddings_line_first.test_filter_urls(triple_list_hdbscan_line_first)
        self.__metrics_hdbscan_line_second = self.__embeddings_line_second.test_filter_urls(triple_list_hdbscan_line_second)

    def plot_normalized_data_normal_skipgram(self, file_name="normalized_data_normal_skipgram_"):
        return self.__embeddings_normal_skipgram.plot_normalized_data(file_name + self.__config_normal_skipgram)

    def plot_normalized_data_line_first(self, file_name="normalized_data_line_first"):
        return self.__embeddings_line_first.plot_normalized_data(file_name)

    def plot_normalized_data_line_second(self, file_name="normalized_data_line_second"):
        return self.__embeddings_line_second.plot_normalized_data(file_name)

    def get_dataframe_normal_skipgram(self):
        return F.get_dataframe_metrics(self.__metrics_kmeans_normal_skipgram, self.__metrics_hdbscan_normal_skipgram)

    def get_dataframe_line_first(self):
        return F.get_dataframe_metrics(self.__metrics_kmeans_line_first, self.__metrics_hdbscan_line_first)

    def get_dataframe_line_second(self):
        return F.get_dataframe_metrics(self.__metrics_kmeans_line_second, self.__metrics_hdbscan_line_second)