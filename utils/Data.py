import numpy as np

from utils.UrlMap import UrlMap


class Data:

    def __init__(self, file_path, url_map=None):
        assert isinstance(file_path, str), "file_path must be a string"
        assert url_map is None or isinstance(url_map, UrlMap), "url_map must be an UrlMap object or None"

        mapCode = isinstance(url_map, UrlMap)

        self.embeddings = np.zeros((1, 1))
        self.map = dict()
        self.reverseMap = dict()

        number_line = 0
        for line in open(file_path):
            line = line.replace("\n", "")

            words = line.split(" ")
            if number_line is 0:
                rows = int(words[0])
                columns = int(words[1])

                self.embeddings = np.zeros((rows, columns))
            else:
                id = words[0]
                pos = number_line - 1

                if mapCode:
                    url = url_map.get_url(id)

                    self.map[url] = pos
                    self.reverseMap[pos] = url
                else:
                    self.map[id] = pos
                    self.reverseMap[pos] = id

                for index in range(1, columns+1):
                    value = np.float64(words[index])
                    self.embeddings[pos, index - 1] = value
            number_line += 1

    @property
    def get_pos(self, key):
        return self.map[key]

    @property
    def get(self, position):
        return self.reverseMap[position]

    @property
    def get_embeddings(self):
        return self.embeddings

    @property
    def get_words(self):
        return [self.reverseMap[pos] for pos in range(0, len(self.reverseMap))]