import os.path

import numpy as np


class UrlConverter:

    def __init__(self, file_url_clusterLabel, file_url_codeUrl, separator):
        assert os.path.isfile(file_url_clusterLabel), "the file %r does not exist" % file_url_clusterLabel
        assert os.path.isfile(file_url_codeUrl), "the file %r does not exist" % file_url_codeUrl
        self.__map_codeUrl_clusteringLabel = self.__generate_map_codeUrl_label(file_url_clusterLabel, file_url_codeUrl, separator)

    def __generate_map_codeUrl_label(self, file_url_clusterLabel, file_url_codeUrl, separator):
        in_file1 = open(file_url_clusterLabel, "r")
        text = in_file1.readlines()
        #map1 is the groundtruth
        map1 = {line.rstrip().split(separator)[0]:line.rstrip().split(separator)[1] for line in text}
        in_file1.close()

        in_file2 = open(file_url_codeUrl, "r")
        text = in_file2.readlines()
        #map2 is the {url, code}
        map2 = {line.rstrip().split(separator)[0]:line.rstrip().split(separator)[1] for line in text}
        in_file2.close()

        map_code_label = {}
        #for url, code in map2.items():
        #    label = map1[url]
        #    map_code_label[code] = int(label)
        for url, label in map1.items():
            code = map2[url]
            map_code_label[code] = int(label)

        return map_code_label

    def get_ordered_labels(self, list_codes_url):
        assert isinstance(list_codes_url, list) or isinstance(list_codes_url, np.ndarray), "list_codes_url must be a list or a numpy array"

        res = [self.__map_codeUrl_clusteringLabel[id_url] for id_url in list_codes_url]
        return np.array(res)

    def get_triple_list(self, list_codes_url, learned_labels):
        assert isinstance(list_codes_url, list) or isinstance(list_codes_url, np.ndarray), "list_codes_url must be a list or a numpy array"
        assert isinstance(learned_labels, list) or isinstance(learned_labels, np.ndarray), "learned_labels must be a list or a numpy array"
        assert len(list_codes_url) == len(learned_labels), "list_codes_url and learned_labels must be the same length"

        result = []
        for i in range(0, len(list_codes_url)):
            code_url = list_codes_url[i]
            learned_label = learned_labels[i]
            real_label = self.__map_codeUrl_clusteringLabel[code_url]

            result.append((code_url, real_label, learned_label))

        return np.array(result)

    def print_url_converter(self):
        for k, v in self.__map_codeUrl_clusteringLabel:
            print(k, v)

    @property
    def get_map(self):
        return self.__map_codeUrl_clusteringLabel

    @property
    def get_true_clusteringLabels(self):
        return [self.__map_codeUrl_clusteringLabel[code_url] for code_url in self.__map_codeUrl_clusteringLabel.keys()]