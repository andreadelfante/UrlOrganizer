import numpy as np
from utils.UrlsEmbedding import UrlsEmbedding

from models.UrlConverter import UrlConverter


def main(args):
    direct = "/Volumes/AdditionalDriveMAC/Google Drive/1) Tesi/Sperimentazioni/" \
             "cs.ox.ac.uk/" \
             "NoConstraint/" \
             "cs.ox.ac.uk.NoConstraint.words100000.depth10.window2.iteractions50/"
    clustering = "kmeans"
    scaling = "l2"

    file_url_codeUrl = direct + "seedsMap.txt"
    file_url_cluster = direct + "groundTruth.csv"
    file_embeddings = direct + "embeddings_with_b.txt"
    file_embeddings_normal = direct + "embeddings_normal.txt"
    separator = "\t"

    converter = UrlConverter(file_url_cluster, file_url_codeUrl, separator)
    left_embeddings = UrlsEmbedding(file_embeddings, scaling)
    normal_embeddings = UrlsEmbedding(file_embeddings_normal, scaling)

    normal_embeddings.intersect(left_embeddings.get_urls)
    left_embeddings.intersect(normal_embeddings.get_urls)

    labels = left_embeddings.clustering()
    result = converter.get_triple_list(left_embeddings.get_urls, labels)

    print(np.array_equiv(normal_embeddings.get_urls, left_embeddings.get_urls))

    #converter.get_triple_list(normal_embeddings.get_urls, normal_embeddings.)

    print("done")

if __name__ == "__main__":
    main([])