import sys

from utils.UrlConverter import UrlConverter
from utils.UrlsEmbedding import UrlsEmbedding, Scale


def main(argv):
    if len(argv) != 4:
        print('Wrong number of arguments. Inserted ', len(sys.argv), ', required 3')
        print('UrlConverted.py <filename hashmap (url -> code)> <filename hashmap (url -> cluster_label)> <separator>')
        print('UrlConverted /path/to/urlsMap.txt /path/to/urlToMembership.txt ,')
        sys.exit(2)
    converter = UrlConverter(argv[0], argv[1], argv[2])
    embeddings = UrlsEmbedding(file_path=argv[3], scaling=Scale.minmax)
    learned_labels = embeddings.clustering()
    embeddings.test(url_converter=converter, learned_labels=learned_labels)

if __name__ == "__main__":
    # argv = sys.argv[1:]
    # main(argv)
    file_url_cluster = "dataset/illinois/ground_truth/groundTruth.txt"
    file_url_codeUrl = "dataset/illinois/list_constraint/urlMap.txt"
    separator = ","
    file_embeddings = "dataset/illinois/list_constraint/normalSkipgram.txt"
    main(argv=[file_url_cluster, file_url_codeUrl, separator, file_embeddings])