from sklearn import metrics
import pandas as pd

class Metrics:

    def __init__(self, columns):
        self.columns = columns
        self.index = []
        self.data = []

    def addRow(self, ground_truth, labels, embeddings, index, metric="euclidean"):
        self.data.append([
            metrics.homogeneity_score(ground_truth, labels),
            metrics.completeness_score(ground_truth, labels),
            metrics.v_measure_score(ground_truth, labels),
            metrics.adjusted_rand_score(ground_truth, labels),
            metrics.adjusted_mutual_info_score(ground_truth, labels),
            metrics.silhouette_score(embeddings, labels, metric='euclidean')
        ])

        self.index.append(index)

    def show(self):
        metrics = pd.DataFrame(data=self.data, index=self.index, columns=self.columns)
        print(metrics)