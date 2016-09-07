import unittest

import numpy as np
import numpy.testing as np_testing

from models.UrlsEmbedding import UrlsEmbedding, Scale


class UrlsEmbeddingTest(unittest.TestCase):

    def setUp(self):
        self.file_paths = [
            "dataset/embeddings_first.txt",
            "dataset/embeddings_second.txt",
            "dataset/embeddings_third.txt"
        ]

        self.types = [
            float,
            int,
            None
        ]

        self.file_path_for_intersection = "dataset/embeddings_for_intersection.txt"

        self.other_urls = [
                np.array([]),
                np.array(['0', '5', '8']),
                np.array(['0', '1', '5', '7', '8'])
            ]

        self.intersect_results = [
            np.array([]),
            np.array(['0', '5', '8']),
            np.array(['0', '1', '5', '7', '8']),
        ]

    def tearDown(self):
        self.file_paths = None
        self.types = None
        self.place_urls = None
        self.other_urls = None
        self.intersect_results = None
        self.file_path_for_intersection = None

    def test_scale(self):
        file_path = self.file_paths[1]
        urls_embedding_zscore = UrlsEmbedding(file_path=file_path, scaling=Scale.zscore)
        urls_embedding_minmax = UrlsEmbedding(file_path=file_path, scaling=Scale.minmax)
        urls_embedding_none = UrlsEmbedding(file_path=file_path, scaling=Scale.none)

        original_embedding_none = urls_embedding_none.get_original_embedding
        scaled_embedding_none = urls_embedding_none.get_scaled_embeddings

        np_testing.assert_array_equal(
            original_embedding_none,
            scaled_embedding_none,
            "scaling none are failed"
        )

        original_embedding_minmax = urls_embedding_minmax.get_original_embedding
        scaled_embedding_minmax = urls_embedding_minmax.get_scaled_embeddings
        expected_minmax = self.__minmax(original_embedding_minmax)

        np_testing.assert_array_equal(
            scaled_embedding_minmax,
            expected_minmax,
            "minmax failed",
        )

        original_embedding_zscore = urls_embedding_zscore.get_original_embedding
        scaled_embedding_zscore = urls_embedding_zscore.get_scaled_embeddings
        expected_zscore = self.__z_score(original_embedding_zscore)

        np_testing.assert_array_equal(
            scaled_embedding_zscore,
            expected_zscore,
            "zscore failed",
        )

    def test_read_embeddings(self):
        assert len(self.file_paths) == len(self.types), "file paths and types lengths are not the same"

        for i in range(len(self.file_paths)):
            urls_embedding = UrlsEmbedding(file_path=self.file_paths[i])
            result_urls = urls_embedding.get_urls
            result_embeddings = urls_embedding.get_original_embedding.astype(self.types[i])

            file = open(self.file_paths[i])
            lines = file.readlines()

            self.assertEqual(first=len(result_urls), second=len(result_embeddings),
                             msg="result_urls and result_embeddings are not the same length")
            self.assertEqual(first=len(result_urls), second=len(lines),
                             msg="result_urls and lines are not the same length")

            for i in range(len(lines)):
                line = lines[i].replace("\n", "")
                url = result_urls[i]
                embedding = result_embeddings[i, :]
                result_line = self.__concatenate(url, embedding)

                self.assertEqual(first=line, second=result_line,
                                 msg="the first line and the second line are not the same")

            file.close()

    def test_intersect(self):
        assert len(self.other_urls) == len(self.intersect_results), "other urls and intersect results " \
                                                                    "must be the same lengths"
        for i in range(len(self.other_urls)):
            intersect = UrlsEmbedding(self.file_path_for_intersection, Scale.l2)
            intersect.intersect(self.other_urls[i])

            expected_result = self.intersect_results[i]
            np_testing.assert_array_equal(intersect.get_urls, expected_result,
                                          err_msg="urls and intersect_results must be the same")

    def __concatenate(self, id_url, embeddings):
        result = ""
        for element in embeddings:
            result += str(element) + " "

        result = str(id_url) + " " + result

        return result.strip()

    def __minmax(self, embeddings):
        newVector = embeddings.copy()

        for index_column in range(0, len(embeddings[0])):
            column = embeddings[:, index_column]

            min = column.min()
            max = column.max()

            for index_row in range(0, len(column)):
                value = column[index_row]
                newVector[index_row, index_column] = (value - min) / (max - min)

        return newVector

    def __z_score(self, embeddings):
        newVector = embeddings.copy()

        for index_column in range(0, len(embeddings[0])):
            row = embeddings[:, index_column]

            mean = row.mean()
            standard_deviation = row.std()

            for index_row in range(0, len(row)):
                value = row[index_row]
                newVector[index_row, index_column] = (value - mean) / (standard_deviation)

        return newVector


if __name__ == '__main__':
    unittest.main()
