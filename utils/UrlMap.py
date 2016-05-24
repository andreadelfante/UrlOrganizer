class UrlMap:

    def __init__(self, file_path, separator=","):
        self.map = dict([s.strip() for s in line.split(separator)] for line in open(file_path))

    def get_id(self, url):
        ret = ""
        try:
            ret = self.map[url]
        except KeyError:
            print("Url not found")
        return ret

    def __getitem__(self, item):
        return self.get_id(id=item)

    def get_url(self, id):
        for key in self.map.values():
            if key == id:
                return key

        raise KeyError

    def contains_id(self, url):
        try:
            self.get_id(url)
            return True
        except:
            return False

    def contains_url(self, id):
        try:
            self.get_url(id)
            return True
        except:
            return False

    def __iter__(self):
        return self.map.__iter__()

    def remove_elements_not_found(self, another_set):
        assert isinstance(another_set, set), "another_set must be a set"

        url_not_in_sequences = list(self.key_set - another_set)
        print(len(url_not_in_sequences))
        for url in url_not_in_sequences:
            del self.map[url]

    @property
    def key_set(self):
        return set(self.map)