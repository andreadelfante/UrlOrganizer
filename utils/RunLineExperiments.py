from models.UrlsEmbedding import UrlsEmbedding

import utils.Formatter as F
from models.UrlConverter import UrlConverter


class RunLineExperiments:

    def __init__(self, direct, prefix,
                 db_left_skipgram,
                 window_left_skipgram,
                 depth_left_skipgram,
                 iteractions_left_skipgram,
                 db_normal_skipgram,
                 window_normal_skipgram,
                 depth_normal_skipgram,
                 iteractions_normal_skipgram,
                 clustering,
                 separator="\\t", scale="none", intersect=False):
        config_normal_skipgram = "words" + db_normal_skipgram + \
                                 ".depth" + depth_normal_skipgram + \
                                 ".window" + window_normal_skipgram + \
                                 ".iteractions" + iteractions_normal_skipgram

        config_left_skipgram = "words" + db_left_skipgram + \
                                 ".depth" + depth_left_skipgram + \
                                 ".window" + window_left_skipgram + \
                                 ".iteractions" + iteractions_left_skipgram

        direct_normal_skipgram = direct + prefix + config_normal_skipgram + "/"
        direct_left_skipgram = direct + prefix + config_left_skipgram + "/"

        file_url_codeUrl = direct_normal_skipgram + "seedsMap.txt"
        file_url_cluster = direct_normal_skipgram + "groundTruth.csv"
        file_embeddings_normal_skipgram = direct_normal_skipgram + "embeddings_normal.txt"
        file_embeddings_left_skipgram = direct_left_skipgram + "embeddings_with_b.txt"
        file_embeddings_line_first = direct + "embeddings_line_first.txt"
        file_embeddings_line_second = direct + "embeddings_line_second.txt"

        converter = UrlConverter(file_url_cluster, file_url_codeUrl, separator)
        self.__embeddings_normal_skipgram = UrlsEmbedding.init_from_embeddings(file_embeddings_normal_skipgram, scale)
        self.__embeddings_left_skipgram = UrlsEmbedding.init_from_embeddings(file_embeddings_left_skipgram, scale)
        self.__embeddings_line_first = UrlsEmbedding.init_from_embeddings(file_embeddings_line_first, scale)
        self.__embeddings_line_second = UrlsEmbedding.init_from_embeddings(file_embeddings_line_second, scale)

        if intersect:
            self.__embeddings_normal_skipgram.intersect(self.__embeddings_left_skipgram.get_urls)
            self.__embeddings_left_skipgram.intersect(self.__embeddings_normal_skipgram.get_urls)
            self.__embeddings_line_first.intersect(self.__embeddings_normal_skipgram.get_urls)
            self.__embeddings_line_second.intersect(self.__embeddings_normal_skipgram.get_urls)

        true_labels = converter.get_true_clusteringLabels
        cluster_size = len(set(true_labels))

        learned_labels_line_first = self.__embeddings_line_first.clustering(type_clustering=clustering,
                                                                             n_clusters=cluster_size)
        learned_labels_line_second = self.__embeddings_line_second.clustering(type_clustering=clustering,
                                                                                     n_clusters=cluster_size)

        url_codes_line_first = self.__embeddings_line_first.get_urls
        url_codes_line_second = self.__embeddings_line_second.get_urls

        triple_list_line_first = converter.get_triple_list(list_codes_url=url_codes_line_first,
                                                            learned_labels=learned_labels_line_first)
        triple_list_line_second = converter.get_triple_list(list_codes_url=url_codes_line_second,
                                                                   learned_labels=learned_labels_line_second)

        self.__metrics_line_first = self.__embeddings_line_first.test_filter_urls(triple_list_line_first)
        self.__metrics_line_second = self.__embeddings_line_second.test_filter_urls(triple_list_line_second)

        self.__clustering = clustering

    def plot_normalized_data_line_first(self, file_name="normalized_data_line_first"):
        return self.__embeddings_line_first.plot_normalized_data(file_name)

    def plot_normalized_data_line_second(self, file_name="normalized_data_line_second"):
        return self.__embeddings_line_second.plot_normalized_data(file_name)

    def get_dataframe_line_first(self):
        return F.get_dataframe_metrics_just_one(self.__metrics_line_first, self.__clustering)

    def get_dataframe_line_second(self):
        return F.get_dataframe_metrics_just_one(self.__metrics_line_second, self.__clustering)