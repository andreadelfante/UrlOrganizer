from utils.UrlConverter import UrlConverter
from utils.UrlsEmbedding import UrlsEmbedding, Scale, Clustering_algorithm

class RunExperiments:

    def __init__(self, direct, separator="\t",
                 clustering_algorithm="kmeans", scale="none"):

        file_url_codeUrl = direct + "urlMap.txt"
        file_url_cluster = direct + "groundTruth.csv"
        file_embeddings = direct + "embeddings.txt"

        self.__converter = UrlConverter(file_url_cluster, file_url_codeUrl, separator)
        self.__embeddings = UrlsEmbedding(file_embeddings, scaling=scale)

        true_labels = self.__converter.get_true_clusteringLabels
        learned_labels = self.__embeddings.clustering(type_clustering=clustering_algorithm,
                                                      n_clusters=len(set(true_labels)))

        url_codes = self.__embeddings.get_urls
        triple_list = self.__converter.get_triple_list(list_codes_url=url_codes, learned_labels=learned_labels)

        self.__metrics = self.__embeddings.test_filter_urls(triple_list=triple_list)

    @property
    def get_metrics(self):
        return self.__metrics