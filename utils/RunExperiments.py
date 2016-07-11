from utils.UrlConverter import UrlConverter
from utils.UrlsEmbedding import UrlsEmbedding, Scale, Clustering_algorithm

class RunExperiments:

    def __init__(self, direct, separator="\\t",
                 clustering_algorithm="kmeans", scale="none"):

        file_url_codeUrl = direct + "seedsMap.txt"
        file_url_cluster = direct + "groundTruth.csv"
        file_embeddings_with_b = direct + "embeddings_with_b.txt"
        file_embeddings_no_b = direct + "embeddings_no_b.txt"

        converter = UrlConverter(file_url_cluster, file_url_codeUrl, separator)
        embeddings_with_b = UrlsEmbedding(file_embeddings_with_b, scaling=scale)
        embeddings_no_b = UrlsEmbedding(file_embeddings_no_b, scaling=scale)

        true_labels = converter.get_true_clusteringLabels
        cluster_size = len(set(true_labels))

        learned_labels_with_b = embeddings_with_b.clustering(type_clustering=clustering_algorithm,
                                                                  n_clusters=cluster_size)
        learned_labels_no_b = embeddings_no_b.clustering(type_clustering=clustering_algorithm,
                                                              n_clusters=cluster_size)

        url_codes_with_b = embeddings_with_b.get_urls
        url_codes_no_b = embeddings_no_b.get_urls

        triple_list_with_b = converter.get_triple_list(list_codes_url=url_codes_with_b,
                                                            learned_labels=learned_labels_with_b)
        triple_list_no_b = converter.get_triple_list(list_codes_url=url_codes_no_b,
                                                          learned_labels=learned_labels_no_b)

        self.__metrics_with_b = embeddings_with_b.test_filter_urls(triple_list=triple_list_with_b)
        self.__metrics_no_b = embeddings_no_b.test_filter_urls(triple_list=triple_list_no_b)

    @property
    def get_metrics_with_b(self):
        return self.__metrics_with_b

    @property
    def get_metrics_no_b(self):
        return self.__metrics_no_b