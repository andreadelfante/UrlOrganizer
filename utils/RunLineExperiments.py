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
        file_embeddings_line = direct + "embeddings_line.txt"

        converter = UrlConverter(file_url_cluster, file_url_codeUrl, separator)
        self.__embeddings_normal_skipgram = UrlsEmbedding(file_embeddings_normal_skipgram, scale)
        self.__embeddings_line = UrlsEmbedding(file_embeddings_line, scale)

        if intersect:
            self.__embeddings_line.intersect(self.__embeddings_normal_skipgram.get_urls)
            self.__embeddings_normal_skipgram.intersect(self.__embeddings_line.get_urls)

        true_labels = converter.get_true_clusteringLabels
        cluster_size = len(set(true_labels))

        learned_labels_kmeans_normal_skipgram = self.__embeddings_normal_skipgram.clustering(type_clustering="kmeans",
                                                                                             n_clusters=cluster_size)
        learned_labels_kmeans_line = self.__embeddings_line.clustering(type_clustering="kmeans",
                                                                       n_clusters=cluster_size)
        learned_labels_hdbscan_normal_skipgram = self.__embeddings_normal_skipgram.clustering(type_clustering="hdbscan")
        learned_labels_hdbscan_line = self.__embeddings_line.clustering(type_clustering="hdbscan")

        url_codes_normal_skipgram = self.__embeddings_normal_skipgram.get_urls
        url_codes_line = self.__embeddings_line.get_urls

        triple_list_kmeans_normal_skipgram = converter.get_triple_list(list_codes_url=url_codes_normal_skipgram,
                                                                       learned_labels=
                                                                       learned_labels_kmeans_normal_skipgram)
        triple_list_kmeans_line = converter.get_triple_list(list_codes_url=url_codes_line,
                                                            learned_labels=learned_labels_kmeans_line)
        triple_list_hdbscan_normal_skipgram = converter.get_triple_list(list_codes_url=url_codes_normal_skipgram,
                                                                        learned_labels=
                                                                        learned_labels_hdbscan_normal_skipgram)
        triple_list_hdbscan_line = converter.get_triple_list(list_codes_url=url_codes_line,
                                                             learned_labels=learned_labels_hdbscan_line)

        self.__metrics_kmeans_normal_skipgram = self.__embeddings_normal_skipgram.test_filter_urls(
            triple_list_kmeans_normal_skipgram)
        self.__metrics_kmeans_line = self.__embeddings_line.test_filter_urls(triple_list_kmeans_line)
        self.__metrics_hdbscan_normal_skipgram = self.__embeddings_normal_skipgram.test_filter_urls(
            triple_list_hdbscan_normal_skipgram)
        self.__metrics_hdbscan_line = self.__embeddings_line.test_filter_urls(triple_list_hdbscan_line)

    def plot_normalized_data_normal_skipgram(self, file_name="normalized_data_normal_skipgram_"):
        return self.__embeddings_normal_skipgram.plot_normalized_data(file_name + self.__config_normal_skipgram)

    def plot_normalized_data_line(self, file_name="normalized_data_line"):
        return self.__embeddings_line.plot_normalized_data(file_name)

    def get_dataframe_normal_skipgram(self):
        return F.get_dataframe_metrics(self.__metrics_kmeans_normal_skipgram, self.__metrics_hdbscan_normal_skipgram)

    def get_dataframe_line(self):
        return F.get_dataframe_metrics(self.__metrics_kmeans_line, self.__metrics_hdbscan_line)