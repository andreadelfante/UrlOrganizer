class UrlMap:

    def __init__(self, file_path, separator=","):
        self.map = dict()

        for line in open(file_path):
            words = line.replace("\n", "").split(separator)

            url = words[0]
            id = words[1]

            self.map[id] = url

    def get_url(self, id):
        ret = ""
        try:
            ret = self.map[id]
        except KeyError:
            print("Url not found with key" + str(id))
        return ret

    def __getitem__(self, item):
        return self.get_url(item)

    def __iter__(self):
        return self.map.__iter__()