import sys

from utils.UrlConverter import UrlConverter
from utils.UrlsEmbedding import UrlsEmbedding, Scale, Clustering_algorithm


def main(argv):
    if len(argv) != 4:
        print('Wrong number of arguments. Inserted ', len(sys.argv), ', required 3')
        print('UrlConverted.py <filename hashmap (url -> code)> <filename hashmap (url -> cluster_label)> <separator>')
        print('UrlConverted /path/to/urlsMap.txt /path/to/urlToMembership.txt ,')
        sys.exit(2)
    converter = UrlConverter(argv[0], argv[1], argv[2])
    embeddings = UrlsEmbedding(file_path=argv[3], scaling=Scale.none)

    learned_labels = embeddings.clustering(type_clustering=Clustering_algorithm.KMeans)
    url_codes = embeddings.get_urls

    triple_list = converter.get_triple_list(list_codes_url=url_codes, learned_labels=learned_labels)
    embeddings.test_filter_urls(triple_list=triple_list)

    embeddings.plot_original_data()
    embeddings.plot_scaled_data()

if __name__ == "__main__":
    # argv = sys.argv[1:]
    # main(argv)
    direct = "/Users/Andrea/Desktop/cs.illinois.edu.ListConstraint.words100000.depth10/"
    clustering = "KMEANS"
    scaling = "None"

    file_url_codeUrl = direct + "urlsMap.txt"
    file_url_cluster = direct + "groundTruth.csv"
    file_embeddings = direct + "embeddings_with_b.txt"
    separator = "\t"

    main(argv=[file_url_cluster, file_url_codeUrl, separator, file_embeddings])