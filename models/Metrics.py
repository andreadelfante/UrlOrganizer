class Metrics:

    def __init__(self, homogeneity, completeness, v_measure, adjuster_rand, mutual_information, silhouette):
        self.__homogeneity = homogeneity
        self.__completeness = completeness
        self.__v_measure = v_measure
        self.__adjuster_rand = adjuster_rand
        self.__mutual_information = mutual_information
        self.__silhouette = silhouette

    @property
    def get_homogeneity(self):
        return self.__homogeneity

    @property
    def get_completeness(self):
        return self.__completeness

    @property
    def get_v_measure(self):
        return self.__v_measure

    @property
    def get_adjuster_rand(self):
        return self.__adjuster_rand

    @property
    def get_mutual_information(self):
        return self.__mutual_information

    @property
    def get_silhouette(self):
        return self.__silhouette