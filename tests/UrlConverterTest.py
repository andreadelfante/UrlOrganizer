import unittest

import numpy as np
import numpy.testing as np_testing

from models.UrlConverter import UrlConverter


class UrlConverterTest(unittest.TestCase):

    def setUp(self):
        self.file_paths = [
            ("dataset/ground_truth_first.txt", "dataset/url_map_first.txt"),
            ("dataset/ground_truth_second.txt", "dataset/url_map_second.txt"),
            ("dataset/ground_truth_third.txt", "dataset/url_map_third.txt"),
            ("dataset/ground_truth_fourth.txt", "dataset/url_map_fourth.txt"),
        ]

        self.expected_maps = [
            {'3': -1,
             '4': -1,
             '9': -1,
             '16': 10},
            {'3': -1,
             '4': -1,
             '16': 10},
            {'3': -1,
             '4': -1,
             '9': -1},
            {}
        ]

        self.url_lists = [
            ['3', '4', '16', '9'],
            ['16', '3'],
            ['16'],
            ['9'],
            np.array(['4']),
            [],
            ['-1', '10']
        ]

        self.clusterlabel_lists = [
            np.array([-1, -1, 10, -1]),
            np.array([10, -1]),
            np.array([10]),
            np.array([-1]),
            np.array([-1]),
            np.array([]),
            np.array([])
        ]

        self.learned_label_lists = [
            [1, 2, 3, 4],
            [4, 6],
            [2],
            [4],
            [5],
            [],
            [3, 4]
        ]

        self.triple_lists = [
            np.array([
                ('3', -1, 1),
                ('4', -1, 2),
                ('16', 10, 3),
                ('9', -1, 4)
            ]),
            np.array([
                ('16', 10, 4),
                ('3', -1, 6)
            ]),
            np.array([
                ('16', 10, 2),
            ]),
            np.array([
                ('9', -1, 4),
            ]),
            np.array([
                ('4', -1, 5),
            ]),
            np.array([]),
            np.array([])
        ]

    def tearDown(self):
        self.file_paths = None
        self.expected_maps = None
        self.url_lists = None
        self.clusterlabel_lists = None
        self.learned_label_lists = None
        self.triple_lists = None


    def test_generate_map_codeUrl_label(self):
        assert len(self.file_paths) == len(self.expected_maps), "file_paths and expected_maps have not the same length"

        for i in range(len(self.file_paths)):
            element = self.file_paths[i]
            expected_map = self.expected_maps[i]

            ground_truth = element[0]
            url_map = element[1]

            try:
                converter = UrlConverter(file_url_clusterLabel=ground_truth, file_url_codeUrl=url_map, separator=",")
                result_map = converter.get_map

                self.assertDictEqual(d1=result_map, d2=expected_map, msg="maps are not equal")
            except KeyError:
                self.assertTrue(i==1)

    def test_get_ordered_labels(self):
        assert len(self.file_paths) != 0, "file_paths cannot be empty"
        assert len(self.url_lists) == len(self.clusterlabel_lists), "url_lists and clusterlabel_lists have not the same length"

        element = self.file_paths[0]
        ground_truth = element[0]
        url_map = element[1]

        converter = UrlConverter(file_url_clusterLabel=ground_truth, file_url_codeUrl=url_map, separator=",")

        for i in range(len(self.url_lists)):
            clusterlabel_list_expected = self.clusterlabel_lists[i]
            url_list = self.url_lists[i]

            try:
                result_clusterlabel_list = converter.get_ordered_labels(list_codes_url=url_list)

                np_testing.assert_array_equal(
                    clusterlabel_list_expected,
                    result_clusterlabel_list,
                    "clusterlabel lists are not the same"
                )
            except KeyError:
                self.assertTrue(i==6)

    def test_get_triple_list(self):
        assert len(self.file_paths) != 0, "file_paths cannot be empty"
        assert len(self.learned_label_lists) == len(self.triple_lists), "url_learned_label_lists and triple_lists have not the same length"
        assert len(self.url_lists) == len(self.learned_label_lists), "url_lists and learned_label_lists have not the same length"

        element = self.file_paths[0]
        ground_truth = element[0]
        url_map = element[1]

        converter = UrlConverter(file_url_clusterLabel=ground_truth, file_url_codeUrl=url_map, separator=",")

        for i in range(len(self.learned_label_lists)):
            list_codes = self.url_lists[i]
            learned_labels = self.learned_label_lists[i]
            triple_list_expected = self.triple_lists[i]

            try:
                triple_list_result = converter.get_triple_list(list_codes_url=list_codes, learned_labels=learned_labels)

                np_testing.assert_array_equal(
                    triple_list_result,
                    triple_list_expected,
                    err_msg="triple_list_result and triple_list_expected must be the same"
                )
            except KeyError:
                self.assertTrue(i == 6)