import os
import numpy as np
import sys
from itertools import tee
import gensim
from hdbscan import HDBSCAN
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA
from nltk.stem.snowball import SnowballStemmer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


class Experiment:
    def __init__(self):
        self.vector_dim = 120
        self.piipp = 3

    def get_urlmap(self, filename, sep="\t"):
        return dict( [s.strip() for s in line.split(sep)[::-1]] for line in open(filename, "r"))

    def get_content_map(self, filename, sep="\t"):
        return dict( [s.strip() for s in line.split(sep)] for line in open(filename) )

    def get_sequences(self, filename, sep=" ", min_len=1):
        for line in open(filename, "r").read().splitlines() :
            sequence = line.split(sep)
            if len(sequence) >= min_len:
                yield sequence

    def get_confusion_table(self, ground_truth, predicted_labels):
            assert len(ground_truth) == len(predicted_labels), "Invalid input arguments"
            assert len(ground_truth) > 0, "Invalid input arguments"
            assert isinstance(ground_truth[0], int), "Type is not int"
            assert isinstance(predicted_labels[0], int), "Type is not int"

            # matrix -> ground_truth x predicted_labels
            conf_table = np.zeros((len(set(ground_truth)), len(set(predicted_labels))))
            real_clust = list(set(ground_truth))
            # it's necessary because ground truth can have discontinuous cluster set
            clust_to_index = {real_clust[i]: i for i in range(len(real_clust))}

            for real_clust in clust_to_index.values():
                for i in range(len(predicted_labels)):
                    if clust_to_index[ground_truth[i]] == real_clust:
                        cluster_found = predicted_labels[i]
                        conf_table[real_clust, cluster_found] = conf_table[real_clust, cluster_found] + 1
            return conf_table

    def get_dimension_vectors(self,vectors_type):
        #return pair (dim_link_vector, dim_content_vector)
        if(vectors_type == "link"):
            return (self.vector_dim, 0)
        elif (vectors_type == "content"):
            return (0, self.vector_dim)
        elif (vectors_type == "combined"):
            half = self.vector_dim/2
            return(half, half)
        else:
            print("ERROR you can choose among link, content and combined type")
            exit(2)
    def runWord2Vec(self, word2Vec_conf, dimension_vector):
        if(word2Vec_conf == "negative"):
            print("Link vectors generation using negative sampling")
            return gensim.models.Word2Vec(window = 2, min_count = 0, negative = 5, size = dimension_vector)
        elif(word2Vec_conf == "h_softmax"):
            print("Link vectors generation using hierarchical softmax")
            return gensim.models.Word2Vec(window = 2, min_count = 0, sg = 1, hs = 1, size = dimension_vector)
        elif(word2Vec_conf == "None"):
            print("link vectors generation skipped!!")
        else:
            print("ERROR in word2vec configuration")
            sys.exit(2)

    def get_content_matrix(self, documents, dim_content):
        #create tf-idf vectors
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
             ngram_range = (1, 2))

        tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
        print("Create tf-idf matrix, shape: ", tfidf_matrix.shape)

        #dimensionalty reduction with LSA
        svd = TruncatedSVD(dim_content)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        lsa_tfidf_matrix = lsa.fit_transform(tfidf_matrix)
        print("Dimensionality reduction with lsa, shape: ", tfidf_matrix.shape)

        for i in range (0, len(lsa_tfidf_matrix)):
            for j in range(0, len(lsa_tfidf_matrix[i])):
                if(lsa_tfidf_matrix[i,j]>1):
                    print(lsa_tfidf_matrix[i,j])
        print(lsa_tfidf_matrix)
        return  lsa_tfidf_matrix

    def run(self, working_directory, clustering_algorithm, word2Vec_conf, vector_type):
        vertices_path = working_directory + "vertex.txt"
        seedsMap_path = working_directory + "seedsMap.txt"
        groundTruth =  working_directory + "groundTruth.csv"
        random_walks_path  = working_directory + "sequenceIDs.txt"

        urlsmap = self.get_urlmap(seedsMap_path)
        documents = self.get_content_map(vertices_path)
        groundTruthMap = self.get_content_map(groundTruth)
        random_walks1, random_walks2 = tee(self.get_sequences(random_walks_path))
        #true_labels = np.array([int(groundTruthMap[v]) for v in urlsmap.values()])
        true_labels = [int(groundTruthMap[v]) for v in urlsmap.values()]

        dim_link, dim_content = self.get_dimension_vectors(vector_type)

        embedding_matrix = []
        document_matrix = []
        codes = list(urlsmap.keys())

        if(dim_link>0):
            word2vec = self.runWord2Vec(word2Vec_conf, dim_link)
            word2vec.build_vocab(random_walks1)
            word2vec.train(random_walks2)
            for url in codes:
                embedding = word2vec[url]
                embedding_matrix.append(embedding)

            #Normalize embedding_matrix using L2
            normalizer_embedding = Normalizer(copy=False)
            embedding_matrix = normalizer_embedding.fit_transform(embedding_matrix)
            print("Normalize embedding_matrix, shape: ",embedding_matrix.shape)

        if(dim_content>0):
             for url in codes:
                 document_matrix.append(documents[url])
             content_matrix = self.get_content_matrix(document_matrix, dim_content)

        combined_matrix = []

        if(dim_link>0 and dim_content>0):
            combined_matrix = np.array ([np.concatenate((content_matrix[i], embedding_matrix[i])) for i in range(0, len(content_matrix))])
            print("Combined link and content matrices, shape: ", combined_matrix.shape)
        elif (dim_link>0):
            combined_matrix = embedding_matrix
        else:
            combined_matrix = content_matrix

        #clustering
        if(clustering_algorithm == "KMEANS"):
            num_clusters = len(set(true_labels))
            print("Clustering using KMEANS with num_clusters = ", num_clusters)
            algorithm = KMeans(n_clusters=num_clusters)
        elif (clustering_algorithm == "HDBSCAN"):
            print("Clustering using HDBSCAN with min 5 elements per cluster")
            algorithm = HDBSCAN(min_cluster_size=5)
        else:
            print("ERROR clustering, wrong parameter ", clustering_algorithm)
            sys.exit(2)

        #learned_labels = np.array(map(lambda x: int(x), algorithm.fit_predict(combined_matrix)))
        learned_labels = np.array([int(x) for x in algorithm.fit_predict(combined_matrix.astype(np.float))])

        #metrics analysis
        filtered_true_labels = []
        filtered_learned_labels = []
        filtered_combined_matrix = []
        for i in range(0, len(true_labels)):
            if (true_labels[i] != -1):
                filtered_true_labels.append(true_labels[i])
                filtered_learned_labels.append(learned_labels[i])
                filtered_combined_matrix.append(combined_matrix[i])
        filtered_true_labels = np.array(filtered_true_labels)
        filtered_learned_labels = np.array(filtered_learned_labels)
        filtered_combined_matrix = np.array(filtered_combined_matrix)

        print("Web pages to analyze: ", len(filtered_learned_labels))
        self.homogeneity = metrics.homogeneity_score(filtered_true_labels, filtered_learned_labels)
        self.completeness = metrics.completeness_score(filtered_true_labels,filtered_learned_labels)
        self.v_measure = metrics.v_measure_score(filtered_true_labels, filtered_learned_labels)
        self.ari = metrics.adjusted_rand_score(filtered_true_labels, filtered_learned_labels)
        self.ami = metrics.adjusted_mutual_info_score(filtered_true_labels, filtered_learned_labels)
        self.silhouette = metrics.silhouette_score(filtered_combined_matrix, filtered_learned_labels, metric='cosine')
        print('\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
        print(self.homogeneity, self.completeness, self.v_measure, self.ari, self.ami, self.silhouette)
        return(filtered_true_labels, filtered_learned_labels)

    def main(self,argv):
            if len(argv)!=4:
                print('Wrong number of arguments. Inserted ', len(sys.argv)-1, ', required 4')
                print('run Experiment.py <directory_path> <clustering algorithm (KMEANS or HDBSCAN)> <word2Vec conf (negative, h_softmax, None)> <vectors_type(link, content, combined)>')
                sys.exit(2)
            else:
                directory = argv[0]
                clustering_type = argv[1]
                w2v_type = argv[2]
                vector_type = argv[3]
                self.run(directory, clustering_type, w2v_type,vector_type)



if __name__ == "__main__":
        Experiment().main(sys.argv[1:])