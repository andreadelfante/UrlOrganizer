import numpy as np

class GroundTruth:

    # constructor, needs the file path
    def __init__(self, file_name, sep=","):
        self.ground_truth = dict([s.strip() for s in line.split(sep)] for line in open(file_name))

    # returns the real cluster membership of a URL
    def get_groundtruth(self, url):
        ret = "-1"
        if url.startswith("https"):
            url = url.replace("https", "http")
        if not url.endswith("/"):
            url += "/"
        if url.startswith("http://www."):
            url = url.replace("http://www.", "http://")
        try:
            ret = self.ground_truth[url]
        except KeyError:
            print("cluster not found in ground truth with url " + str(url))
        return ret

    def get_labelset(self):
        return set(self.ground_truth.values())

    def get_clusters(self, words):
        assert isinstance(words, list), "words must be a list"

        return np.array([self.get_groundtruth(url) for url in words])