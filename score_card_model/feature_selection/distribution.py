import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


class Distribution:
    """
      """
    @staticmethod
    def calculate(x: np.ndarray, segment: int = 100)->Tuple[np.ndarray, np.ndarray]:
        min_val = min(x)
        max_val = max(x)
        step = (min_val - max_val) / segment
        Y, X = np.histogram(x, bins=segment, range=(min_val, max_val))
        return Y, X

    @property
    def x(self):
        return self.__x

    @property
    def X(self):
        return self.__X

    @property
    def Y(self):
        return self.__Y

    @property
    def xticks(self):
        return self.__xticks

    @property
    def segment(self)->int:
        return self.__segment

    @segment.setter
    def segment(self, n: int):
        if isinstance(n, int) and n > 0:
            self.__segment = n
        else:
            raise AttributeError("segment must be a positive integer")

    def __call__(self):
        return dict(list(zip(self.xticks, self.Y)))

    def __init__(self, x: np.ndarray, segment: int = 100):
        self.__x = x
        self.segment = segment
        self.calcul_distribution()

    def calcul_distribution(self):
        self.__Y, self.__X = Distribution.calculate(self.x, self.segment)
        self.__xticks = [str(self.X[i]) + '~' + str(self.X[i + 1]) for i in range(len(self.Y))]

    def draw(self):
        xticks = self.xticks
        plt.bar(range(len(self.Y)), self.Y)
        for i, j in zip(range(len(self.Y)), self.Y):
            plt.text(i, j + 0.5, str(float(self.Y[i]) / sum(self.Y) * 100) + "%")

        plt.xlim(-1, self.segment * 1.1)
        plt.xticks(range(len(self.xticks)))
        plt.show()
