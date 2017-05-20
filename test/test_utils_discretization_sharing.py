from unittest import TestCase
import numpy as np
from score_card_model.utils.discretization.sharing import discrete, discrete_features


class TestDiscrete(TestCase):
    @classmethod
    def setUpClass(clz):
        clz.x = np.array([1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11])
        clz.y = y = np.array(["a", "b"])

    def test_default(self):
        result = list(discrete(self.x))
        expected = [1, 2, 2, 3, 1, 3, 4, 5, 1, 3, 4, 5, 5, 5]
        self.assertSequenceEqual(result, expected)

    def test_with_corret_positive_n(self):
        result = list(discrete(self.x, 7))
        expected = [1, 2, 3, 4, 2, 4, 5, 6, 2, 4, 5, 6, 7, 7]
        self.assertSequenceEqual(result, expected)

    def test_with_corret_negative_n(self):
        result = list(discrete(self.x, -9))
        expected = [1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11]
        self.assertSequenceEqual(result, expected)

    def test_with_corret_zero_n(self):
        result = list(discrete(self.x, 0))
        expected = [1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11]
        self.assertSequenceEqual(result, expected)

    def test_with_invalid_n(self):
        with self.assertRaisesRegex(AttributeError, r"n must be a positive int") as a:
            discrete(self.x, "a")

    def test_with_str_array_default(self):
        result = list(discrete(self.y))
        expected = list(self.y)
        self.assertSequenceEqual(result, expected)

    def test_with_str_array_n_zero(self):
        result = list(discrete(self.y, 0))
        expected = list(self.y)
        self.assertSequenceEqual(result, expected)

    def test_with_str_array_positive_n(self):
        result = list(discrete(self.y, 5))
        expected = list(self.y)
        self.assertSequenceEqual(result, expected)


class TestDiscreteFeatures(TestCase):
    @classmethod
    def setUpClass(clz):
        clz.x = np.array([[1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11],
                          [1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11],
                          [1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11]]).T
        clz.y = np.array([["a", "b", "c", "a", "b", "c", "d"],
                          ["a", "b", "c", "a", "b", "c", "d"],
                          ["a", "b", "c", "a", "b", "c", "d"]]).T

    def test_default(self):
        rst = discrete_features(self.x)
        result = list(rst.reshape(rst.shape[0] * rst.shape[1]))
        exp = np.array([[1, 2, 2, 3, 1, 3, 4, 5, 1, 3, 4, 5, 5, 5],
                        [1, 2, 2, 3, 1, 3, 4, 5, 1, 3, 4, 5, 5, 5],
                        [1, 2, 2, 3, 1, 3, 4, 5, 1, 3, 4, 5, 5, 5]]).T
        expected = list(exp.reshape(exp.shape[0] * exp.shape[1]))
        self.assertSequenceEqual(result, expected)

    def test_with_corret_positive_n(self):
        rst = discrete_features(self.x, 7)
        result = list(rst.reshape(rst.shape[0] * rst.shape[1]))
        exp = np.array([[1, 2, 3, 4, 2, 4, 5, 6, 2, 4, 5, 6, 7, 7],
                        [1, 2, 3, 4, 2, 4, 5, 6, 2, 4, 5, 6, 7, 7],
                        [1, 2, 3, 4, 2, 4, 5, 6, 2, 4, 5, 6, 7, 7]]).T
        expected = list(exp.reshape(exp.shape[0] * exp.shape[1]))
        self.assertSequenceEqual(result, expected)

    def test_with_corret_negative_n(self):
        rst = discrete_features(self.x, -9)
        result = list(rst.reshape(rst.shape[0] * rst.shape[1]))
        exp = np.array([[1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11],
                        [1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11],
                        [1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11]]).T
        expected = list(exp.reshape(exp.shape[0] * exp.shape[1]))
        self.assertSequenceEqual(result, expected)

    def test_with_corret_zero_n(self):
        rst = discrete_features(self.x, 0)
        result = list(rst.reshape(rst.shape[0] * rst.shape[1]))
        exp = np.array([[1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11],
                        [1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11],
                        [1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11]]).T
        expected = list(exp.reshape(exp.shape[0] * exp.shape[1]))
        self.assertSequenceEqual(result, expected)

    def test_with_invalid_n(self):
        with self.assertRaisesRegex(AttributeError, r"n must be a positive int") as a:
            discrete_features(self.x, "a")

    def test_with_str_array_default(self):
        rst = discrete_features(self.y)
        result = list(rst.reshape(rst.shape[0] * rst.shape[1]))
        exp = self.y
        expected = list(exp.reshape(exp.shape[0] * exp.shape[1]))
        self.assertSequenceEqual(result, expected)

    def test_with_str_array_zero_n(self):
        rst = discrete_features(self.y, 0)
        result = list(rst.reshape(rst.shape[0] * rst.shape[1]))
        exp = self.y
        expected = list(exp.reshape(exp.shape[0] * exp.shape[1]))
        self.assertSequenceEqual(result, expected)

    def test_with_str_array_positive_n(self):
        rst = discrete_features(self.y, 5)
        result = list(rst.reshape(rst.shape[0] * rst.shape[1]))
        exp = self.y
        expected = list(exp.reshape(exp.shape[0] * exp.shape[1]))
        self.assertSequenceEqual(result, expected)

    def test_with_list_invalid_n(self):
        with self.assertRaisesRegex(AttributeError, r"N must have the same len with the rows' len") as a:
            discrete_features(self.x, [1, 2])

    def test_with_list_positive_n(self):
        rst = discrete_features(self.x, [5, 7, 5])
        result = list(rst.reshape(rst.shape[0] * rst.shape[1]))
        exp = np.array([[1, 2, 2, 3, 1, 3, 4, 5, 1, 3, 4, 5, 5, 5],
                        [1, 2, 3, 4, 2, 4, 5, 6, 2, 4, 5, 6, 7, 7],
                        [1, 2, 2, 3, 1, 3, 4, 5, 1, 3, 4, 5, 5, 5]]).T
        expected = list(exp.reshape(exp.shape[0] * exp.shape[1]))
        self.assertSequenceEqual(result, expected)

    def test_with_list_negative_n(self):
        rst = discrete_features(self.x, [-9, -9, -9])
        result = list(rst.reshape(rst.shape[0] * rst.shape[1]))
        exp = np.array([[1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11],
                        [1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11],
                        [1, 4, 5, 6, 3, 6, 7, 8, 3, 6, 7, 8, 9, 11]]).T
        expected = list(exp.reshape(exp.shape[0] * exp.shape[1]))
        self.assertSequenceEqual(result, expected)
