class UrlMap:

    def __init__(self, file_path, separator=","):
        self.map = dict([s.strip() for s in line.split(separator)] for line in open(file_path))

    def get_url(self, id):
        ret = ""
        try:
            ret = self.map[id]
        except KeyError:
            print("Url not found")
        return ret