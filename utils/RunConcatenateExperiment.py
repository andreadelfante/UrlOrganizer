from models.UrlConverter import UrlConverter
from models.UrlsEmbedding import UrlsEmbedding

import utils.Formatter as F

class RunConcatenateExperiment:

    def __init__(self,
                 direct,
                 site,
                 type_site,
                 db_best_left_with_b,
                 window_best_left_with_b,
                 depth_best_left_with_b,
                 iteractions_left_with_b,
                 db_best_normal,
                 window_best_normal,
                 depth_best_normal,
                 iteractions_normal,
                 clustering,
                 separator="\\t",
                 scale="none",
                 intersect=False):
        direct = direct + site + "/" + type_site + "/"

        file_url_codeUrl = direct + "seedsMap.txt"
        file_url_cluster = direct + "groundTruth.csv"

        file_embeddings_left_with_b = direct + site + "." + type_site + \
                                      ".words" + db_best_left_with_b + \
                                      ".depth" + depth_best_left_with_b + \
                                      ".window" + window_best_left_with_b + \
                                      ".iteractions" + iteractions_left_with_b + "/embeddings_with_b.txt"
        file_embeddings_normal = direct + site + "." + type_site + \
                                 ".words" + db_best_normal + \
                                 ".depth" + depth_best_normal + \
                                 ".window" + window_best_normal + \
                                 ".iteractions" + iteractions_normal + "/embeddings_normal.txt"
        file_embeddings_doc2vec = direct + "embeddings_doc2vec.txt"

        converter = UrlConverter(file_url_cluster, file_url_codeUrl, separator)
        self.__embeddings_left_with_b = UrlsEmbedding(file_embeddings_left_with_b, scaling=scale)
        self.__embeddings_normal = UrlsEmbedding(file_embeddings_normal, scaling=scale)
        self.__embeddings_doc2vec = UrlsEmbedding(file_embeddings_doc2vec, scaling=scale)

        if intersect:
            print("Intersecting...")
            self.__embeddings_normal.intersect(self.__embeddings_left_with_b.get_urls)
            self.__embeddings_left_with_b.intersect(self.__embeddings_normal.get_urls)

        print("Concatenating...")
        self.__embeddings_left_with_b.concatenate(self.__embeddings_doc2vec)
        self.__embeddings_normal.concatenate(self.__embeddings_doc2vec)

        true_labels = converter.get_true_clusteringLabels
        cluster_size = len(set(true_labels))

        learned_labels_left_with_b = self.__embeddings_left_with_b.clustering(type_clustering=clustering,
                                                                           n_clusters=cluster_size)
        learned_labels_normal = self.__embeddings_normal.clustering(type_clustering=clustering,
                                                                           n_clusters=cluster_size)

        url_codes_left_with_b = self.__embeddings_left_with_b.get_urls
        url_codes_normal = self.__embeddings_normal.get_urls

        triple_list_left_with_b = converter.get_triple_list(list_codes_url=url_codes_left_with_b,
                                                            learned_labels=learned_labels_left_with_b)
        triple_list_normal = converter.get_triple_list(list_codes_url=url_codes_normal,
                                                       learned_labels=learned_labels_normal)

        self.__metrics_left_with_b = self.__embeddings_left_with_b.test_filter_urls(triple_list_left_with_b)
        self.__metrics_normal = self.__embeddings_normal.test_filter_urls(triple_list_normal)
        self.__clustering = clustering

    def get_dataframe_left_with_b_doc2vec(self):
        return F.get_dataframe_metrics_just_one(self.__metrics_left_with_b, self.__clustering)

    def get_dataframe_normal_doc2vec(self):
        return F.get_dataframe_metrics_just_one(self.__metrics_normal, self.__clustering)

    def plot_normalized_normal_doc2vec(self, file_name):
        return self.__embeddings_normal.plot_normalized_data(file_name)

    def plot_normalized_left_with_b_doc2vec(self, file_name):
        return self.__embeddings_left_with_b.plot_normalized_data(file_name)