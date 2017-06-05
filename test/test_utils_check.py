from unittest import TestCase
import numpy as np
from ScoreCardModel.utils.check import check_array_binary, check_array_continuous


class TestCheckArrayBinary(TestCase):
    @classmethod
    def setUpClass(clz):
        clz.x = np.array([1, 4, 5, 6, 3, 9])
        clz.y = np.array(["a", "b", "a", "b", "a", "b", "c"])

        clz.z = np.array([True, False, True, False, True, False])
        clz.m = np.array([1, 0, 0, 1, 0, 1])
        clz.n = np.array(["a", "b", "a", "b", "a", "b"])
        clz.o = np.array([1, 2, 2, 1, 2, 1])

    def test_int(self):
        self.assertFalse(check_array_binary(self.x))

    def test_str(self):
        self.assertFalse(check_array_binary(self.y))

    def test_bool(self):
        self.assertTrue(check_array_binary(self.z))

    def test_01(self):
        self.assertTrue(check_array_binary(self.m))

    def test_2strvalue(self):
        self.assertTrue(check_array_binary(self.n))

    def test_2intvalue(self):
        self.assertTrue(check_array_binary(self.o))


class TestCheckArrayContinuous(TestCase):
    @classmethod
    def setUpClass(clz):
        clz.x = np.array([1, 4, 5, 6, 3, 9])
        clz.y = np.array(["a", "b", "a", "b", "a", "b", "c"])
        clz.z = np.array([True, False, True, False, True, False])

        clz.m = np.array([1.1, 4.1, 5.1, 6.1, 3.1, 9.1])

    def test_int(self):
        self.assertFalse(check_array_continuous(self.x))

    def test_str(self):
        self.assertFalse(check_array_continuous(self.y))

    def test_bool(self):
        self.assertFalse(check_array_continuous(self.z))

    def test_float(self):
        self.assertTrue(check_array_continuous(self.m))
