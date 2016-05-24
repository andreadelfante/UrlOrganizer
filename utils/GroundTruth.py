import os

from utils import UrlMap


class GroundTruth:
    file_path = os.path.abspath(os.path.dirname(__file__)) + "/../../dataset/ground_truth/urlToMembership.txt"

    # constructor, needs the file path
    def __init__(self, file_name=file_path, sep=",", url_map=None):
        if url_map is None:
            assert isinstance(url_map, UrlMap) or isinstance(url_map, dict), "url_map must be an UrlMap object or dict"

        self.ground_truth = dict([s.strip() for s in line.split(sep)] for line in open(file_name))

        self.clusters = [int(self.get_groundtruth(url_map[key])) for key in url_map]

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

    @property
    def get_clusters(self):
        return self.clusters