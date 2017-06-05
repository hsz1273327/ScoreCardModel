from unittest import TestCase
import numpy as np
from ScoreCardModel.feature_selection.distribution import Distribution


class TestDistribution(TestCase):
    @classmethod
    def setUpClass(clz):
        clz.x = np.array([91, 16, 43, 13, 81, 91, 90,  4, 28, 63, 87, 36, 33, 96, 23, 74, 64,
                          33, 84, 99, 89, 48,  9,  8, 34, 17, 31, 39, 65, 98, 90, 36, 41, 57,
                          29, 83, 79, 85, 56, 40, 12, 95, 59, 26, 38, 88,  5, 53, 20, 86, 50,
                          36, 74, 51, 10, 63, 17, 11, 47, 15, 47, 88, 34, 46,  1, 52, 71, 22,
                          49, 67, 93, 11, 71,  2, 90, 68, 25, 52,  3, 79, 50, 33, 35,  2, 42,
                          37, 69, 24, 52, 29, 43, 38, 71, 99, 50, 41, 84, 27, 80, 17])
        clz.d = Distribution(clz.x, 10)
        clz.X = [round(i, 4) for i in np.array([1.,  10.8,  20.6,  30.4,  40.2,  50.,  59.8,  69.6,  79.4,
                                                89.2,  99.])]
        clz.Y = list(np.array([9, 10,  9, 15, 10, 11,  7,  7, 11, 11]))
        clz.xticks = ['1.0~10.8',
                      '10.8~20.6',
                      '20.6~30.4',
                      '30.4~40.2',
                      '40.2~50.0',
                      '50.0~59.8',
                      '59.8~69.6',
                      '69.6~79.4',
                      '79.4~89.2',
                      '89.2~99.0']

        clz.call = {'1.0~10.8': 9,
                    '10.8~20.6': 10,
                    '20.6~30.4': 9,
                    '30.4~40.2': 15,
                    '40.2~50.0': 10,
                    '50.0~59.8': 11,
                    '59.8~69.6': 7,
                    '69.6~79.4': 7,
                    '79.4~89.2': 11,
                    '89.2~99.0': 11}

    def test_segment_invalid_negative(self):
        with self.assertRaisesRegex(AttributeError, r"segment must be a positive integer"):
            Distribution(self.x, -10)

    def test_segment_invalid_str(self):
        with self.assertRaisesRegex(AttributeError, r"segment must be a positive integer"):
            Distribution(self.x, "10")

    def test_default_x(self):
        self.assertSequenceEqual(list(self.x), list(self.d.x))

    def test_default_segment(self):
        self.assertEqual(10, self.d.segment)

    def test_default_xticks(self):
        self.assertSequenceEqual(self.xticks, self.d.xticks)

    def test_default_X(self):
        self.assertSequenceEqual(self.X, [round(i, 4) for i in self.d.X])

    def test_default_Y(self):
        self.assertSequenceEqual(self.Y, list(self.d.Y))

    def test_default_call(self):
        self.assertDictEqual(self.call, self.d())
