import numpy as np

from utils.UrlMap import UrlMap


class Data:

    def __init__(self, file_path, url_map):
        assert isinstance(file_path, str), "file_path must be a string"
        assert isinstance(url_map, UrlMap), "url_map must be an UrlMap object"

        self.embeddings = np.zeros((1, 1))
        self.map_id_position = dict()
        self.map_url_id = dict()

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
                self.map_id_position[id] = number_line-1

                if url_map.contains_url(id):
                    self.map_url_id[url_map.get_url(id)] = id

                for index in range(1, columns+1):
                    value = np.float64(words[index])
                    self.embeddings[number_line - 1, index - 1] = value
            number_line += 1

    def get_index(self, id_url):
        return self.map.get(key=id_url)

    def get_embedding(self, id_url):
        return self.get_embeddings[self.map.get(key=id_url)]

    def get_embeddings(self):
        return self.embeddings

    @property
    def id_url_set(self):
        return set(self.map_url_id)