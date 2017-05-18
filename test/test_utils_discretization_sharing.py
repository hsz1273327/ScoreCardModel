from unittest import TestCase
import numpy as np
from score_card_model.utils.discretization.sharing import discrete, discrete_features


class TestDiscrete(TestCase):
    @classmethod
    def setUpClass(clz):
        clz.x = np.array([1,4,5,6,3,6,7,8,3,6,7,8,9,11])
        clz.y = y = np.array(["a","b"])
    def test_default(self):
        result = list(discrete(self.x))
        expected = [1, 2, 2, 3, 1, 3, 4, 5, 1, 3, 4, 5, 5, 5]
        self.assertSequenceEqual(result,expected)
    def test_with_corret_positive_n(self):
        result = list(discrete(self.x,5))
        expected = [1, 2, 2, 3, 1, 3, 4, 5, 1, 3, 4, 5, 5, 5]
        self.assertSequenceEqual(result,expected)
    def test_with_corret_negative_n(self):
        result = list(discrete(self.x,-9))
        expected = [1,4,5,6,3,6,7,8,3,6,7,8,9,11]
        self.assertSequenceEqual(result,expected)
    def test_with_corret_zero_n(self):
        result = list(discrete(self.x,0))
        expected = [1,4,5,6,3,6,7,8,3,6,7,8,9,11]
        self.assertSequenceEqual(result,expected)
    def test_with_invalid_n(self):
        with self.assertRaisesRegex(AttributeError,r"n must be a int") as a:
            discrete(self.x,"a")
    def test_with_str_array(self):
        result = list(discrete(self.y,0))
        expected = list(self.y)
        self.assertSequenceEqual(result,expected)
