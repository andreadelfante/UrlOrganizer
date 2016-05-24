import os

class GroundTruth:
    filepath = os.path.abspath(os.path.dirname(__file__)) + "/../../dataset/ground_truth/urlToMembership.txt"

    # constructor, needs the file path
    def __init__(self, fpath=filepath, sep=","):
        self.ground_truth = dict([s.strip() for s in line.split(sep)] for line in open(fpath))

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
            print("Url not found")
        return ret

    def get_labelset(self):
        return set(self.ground_truth.values())