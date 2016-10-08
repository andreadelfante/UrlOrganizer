from models.UrlConverter import UrlConverter
from models.UrlsEmbedding import UrlsEmbedding

import utils.Formatter as F

class RunDocumentExperiment:

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
                 use_tfidf=False,
                 separator="\\t",
                 scale="none",
                 intersect=False,
                 dimension_deduction=100):
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
        self.__embeddings_left_with_b = UrlsEmbedding.init_from_embeddings(file_embeddings_left_with_b, scaling=scale)
        self.__embeddings_normal = UrlsEmbedding.init_from_embeddings(file_embeddings_normal, scaling=scale)

        if use_tfidf:
            print("use tfidf")
            self.__embeddings_content = UrlsEmbedding.init_from_vertex(direct + "vertex.txt", dimension_deduction,
                                                                       scale)
        else:
            print("use doc2vec")
            self.__embeddings_content = UrlsEmbedding.init_from_embeddings(file_embeddings_doc2vec, scaling=scale)

        if intersect:
            print("Intersecting...")
            self.__embeddings_normal.intersect(self.__embeddings_left_with_b.get_urls)
            self.__embeddings_left_with_b.intersect(self.__embeddings_normal.get_urls)

            print(str(len(self.__embeddings_normal.get_urls)) + "==" + str(len(self.__embeddings_left_with_b.get_urls)))

            # utilizza solo url in comune tra i due
            self.__embeddings_content.intersect(self.__embeddings_normal.get_urls)
            print("length content: " + str(len(self.__embeddings_content.get_urls)))

        true_labels = converter.get_true_clusteringLabels
        cluster_size = len(set(true_labels))

        learned_labels = self.__embeddings_content.clustering(type_clustering=clustering, n_clusters=cluster_size)

        url_codes_content = self.__embeddings_content.get_urls

        triple_list_content = converter.get_triple_list(url_codes_content, learned_labels)

        self.__metrics_content = self.__embeddings_content.test_filter_urls(triple_list_content)
        self.__clustering = clustering

    def get_dataframe_content(self):
        return F.get_dataframe_metrics_just_one(self.__metrics_content, self.__clustering)

    def plot_normalized_content(self, file_name):
        return self.__embeddings_normal.plot_normalized_data(file_name)