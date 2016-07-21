import sys

from utils.UrlConverter import UrlConverter
from utils.UrlsEmbedding import UrlsEmbedding, Scale, Clustering_algorithm
import numpy as np


def main(args):
    if len(args) != 6:
        print('Wrong number of arguments. Inserted ', len(sys.argv), ', required 3')
        print('UrlConverted.py <filename hashmap (url -> code)> <filename hashmap (url -> cluster_label)> <separator> '
              '<file_embeddings> <scaling> <clustering>')
        sys.exit(2)
    converter = UrlConverter(args[0], args[1], args[2])
    embeddings = UrlsEmbedding(file_path=args[3], scaling=args[4])

    original = embeddings.get_original_embedding
    scaled = embeddings.get_scaled_embeddings
    result = np.array_equal(original, scaled)

    '''learned_labels = embeddings.clustering(type_clustering=args[5])
    url_codes = embeddings.get_urls

    triple_list = converter.get_triple_list(list_codes_url=url_codes, learned_labels=learned_labels)
    embeddings.test_filter_urls(triple_list=triple_list)'''

if __name__ == "__main__":
    # argv = sys.argv[1:]
    # main(argv)
    direct = "/Users/Andrea/Google Drive/1) Tesi/Sperimentazioni" \
             "/cs.illinois.edu.ListConstraint.words100000.depth10.window2.iteractions50/"
    clustering = "kmeans"
    scaling = "l2"

    file_url_codeUrl = direct + "seedsMap.txt"
    file_url_cluster = direct + "groundTruth.csv"
    file_embeddings = direct + "embeddings_with_b.txt"
    separator = "\t"

    main(args=[file_url_cluster, file_url_codeUrl, separator, file_embeddings, scaling, clustering])