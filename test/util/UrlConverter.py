import os.path
import sys

_author_ = 'fabianalanotte'

class UrlConverter:
    def __init__(self, file_url_clusterLabel, file_url_codeUrl, separator):
        assert os.path.isfile(file_url_clusterLabel), "the file %r does not exist" % file_url_clusterLabel
        assert os.path.isfile(file_url_codeUrl), "the file %r does not exist" % file_url_codeUrl
        self.map_codeUrl_clusteringLabel = self.__generate_map_codeUrl_label(file_url_clusterLabel, file_url_codeUrl, separator)

    def __generate_map_codeUrl_label(self, file_url_clusterLabel, file_url_codeUrl, separator):
        in_file1 = open(file_url_clusterLabel, "r")
        text = in_file1.readlines()
        map1 = [{line.rstrip().split(separator)[0]:line.rstrip().split(',')[1] } for line in text]
        in_file1.close()

        in_file2 = open(file_url_codeUrl, "r")
        text = in_file2.readlines()
        map2 = [{line.rstrip().split(separator)[0]:line.rstrip().split(',')[1] } for line in text]
        in_file2.close()

        map_code_label = {}
        for url, code in map2.items():
            label = map1[url]
            map_code_label[code] = label

        return map_code_label

    def get_ordered_labels(self, listUrl):
        res = map(lambda u: self.map_codeUrl_clusteringLabel[u], listUrl)
        return res

    def main(self, argv):
        if len(argv)!=3:
            print('Wrong number of arguments. Inserted ', len(sys.argv), ', required 3')
            print('UrlConverted.py <filename hashmap (url -> code)> <filename hashmap (url -> cluster_label)> <separator>')
            print('UrlConverted /path/to/urlsMap.txt /path/to/urlToMembership.txt ,')
            sys.exit(2)
        converter = UrlConverter(argv[0], argv[1], argv[2])
        for k,v in converter.map_codeUrl_clusteringLabel:
            print(k,v)

    if __name__ == "__main__":
        #argv = sys.argv[1:]
        #main(argv)
        file_url_cluster = ""
        file_url_codeUrl = ""
        separator = ","
        main(argv=[file_url_cluster, file_url_codeUrl, separator])