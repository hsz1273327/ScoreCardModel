from unittest import TestCase
import numpy as np
from score_card_model.utils.count import count_binary


class TestCheckArrayContinuous(TestCase):
    @classmethod
    def setUpClass(clz):
        clz.x = np.array([1, 0, 1, 0, 1, 0, 1])
        clz.y = np.array(["a", "b", "a", "b", "a", "b", "b"])
        clz.z = np.array([True, False, True, False, True, False, True])
        clz.m = np.array([1, 0, 1, 0, 1, 0, 2])

    def test_int(self):
        t, f = count_binary(self.x)
        self.assertTrue(t, 4)
        self.assertTrue(f, 3)

    def test_bool(self):

        t, f = count_binary(self.z)
        self.assertTrue(t, 4)
        self.assertTrue(f, 3)

    def test_str_event(self):
        t, f = count_binary(self.y, "a")
        self.assertTrue(t, 4)
        self.assertTrue(f, 3)

    def test_str_without_event(self):
        with self.assertRaisesRegex(AttributeError, r"need a event"):
            count_binary(self.y)

    def test_not_binary(self):
        with self.assertRaisesRegex(AttributeError, r"array must be a binary array"):
            count_binary(self.m)
