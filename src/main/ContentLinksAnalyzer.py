import os.path

import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from src.main.LinksAnalyzer import LinksAnalyzer, Scale




class ContentLinksAnalyzer(LinksAnalyzer):
     def __init__(self, filename_content, filename_embedding, scaling=Scale.zscore):
        super(ContentLinksAnalyzer, self).__init__(filename_embedding, scaling)
        self.tf_idf = self.__read_content(filename_content)
        #self.content_links = self.__concatenate(self.tf_idf, self.normalized_embeddings)

     def __concatenate(self, tf_idf, normalized_embeddings):
        print(tf_idf.shape, normalized_embeddings.shape)
        return [np.concatenate((tf_idf[i].todense(), normalized_embeddings)) for i in range(tf_idf.shape[0]) ]


     def __read_content(self, filename_content):
        assert os.path.isfile(filename_content), "the file %r does not exist" % filename_content
        in_file = open(filename_content, "r")
        text = in_file.readlines()
        content_map = {}
        for line in text:
            tokens = line.rstrip().split("\t", 1)
            url = tokens[0]
            content = tokens[1]
            content_map[url] = content
        in_file.close()

        content_list = []
        for u in self.urls:
            print(u)
            content = content_map[u]
            content_list.append(content)

        tokenize = lambda text: text.split(" ")
        stem = lambda token, stemmer = SnowballStemmer("english"): stemmer.stem(token)
        tokenize_and_stem = lambda text, stemmer = SnowballStemmer("english"): [stem(token, stemmer) for token in tokenize(text)]


        tfidf_vectorizer = TfidfVectorizer(
            max_df = 0.9,
            max_features = 200000,
            min_df = 0.05,
            stop_words = 'english',
            use_idf = True,
            tokenizer = tokenize_and_stem,
            ngram_range = (1, 3)
        )
        print(content_list)
        tfidf_matrix = tfidf_vectorizer.fit_transform(content_list)
        #svd = TruncatedSVD(n_components = 50, algorithm="arpack", random_state=1)
        #return svd.fit_transform(tfidf_matrix)
        return tfidf_matrix

