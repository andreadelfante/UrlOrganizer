import os

import numpy as np
from nltk import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer


def read_vertex_file(vertex_file_path):
    assert os.path.isfile(vertex_file_path), "the file %r does not exist" % vertex_file_path

    in_file = open(vertex_file_path, "r")
    text = in_file.readlines()
    urls = []
    matrix = []

    for line in text:
        tokens = line.rstrip().split('\t', 1)
        url = tokens[0]
        text = tokens[1]
        matrix.append(text)
        urls.append(url)

    in_file.close()
    return np.array(urls), np.array(matrix)

def fit(document_matrix, embedding_dimension):
    assert isinstance(document_matrix, np.ndarray) or isinstance(document_matrix, list),\
        "document_matrix must be a numpy array or a list"

    if isinstance(document_matrix, np.ndarray):
        document_matrix = document_matrix.tolist()

    tokenize = lambda text: text.split(" ")
    stem = lambda token, stemmer=SnowballStemmer("english"): stemmer.stem(token)
    tokenize_and_stem = lambda text, stemmer=SnowballStemmer("english"): [stem(token, stemmer) for token in
                                                                          tokenize(text)]

    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.9,
        max_features=200000,
        min_df=0.05,
        stop_words='english',
        use_idf=True,
        tokenizer=tokenize_and_stem,
        ngram_range=(1, 2))

    print("Fitting tfidf...")
    tfidf_matrix = tfidf_vectorizer.fit_transform(document_matrix)
    print("Create tf-idf matrix, shape: ", tfidf_matrix.shape)

    svd = TruncatedSVD(embedding_dimension)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    print("Performing lsa dimensional reduction with dimension=" + str(embedding_dimension))
    lsa_tfidf_matrix = lsa.fit_transform(tfidf_matrix)
    print("Dimensionality reduction with lsa, shape: ", lsa_tfidf_matrix.shape)

    return np.array(lsa_tfidf_matrix)

if __name__ == "__main__":
    vertex_file_path = direct = "/Volumes/AdditionalDriveMAC/Google Drive/1) Tesi/Sperimentazioni/" \
                                "cs.illinois.edu/" \
                                "ListConstraint"

    urls, embeddings = read_vertex_file(vertex_file_path + "/vertex.txt")
    result = fit(embeddings, 100)

    print(result)
    print("Done.")