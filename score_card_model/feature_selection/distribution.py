__all__ = ["Distribution"]
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


class Distribution:
    """
    Property:
        x (np.ndarray): - 输入的数组
        X (np.ndarray): - 数组分段的下表
        Y (np.ndarray): - 数组中的元素每个分段中的出现的个数
        xticks (np.ndarray): - 均分的分段信息
        segment (int): - 均分的分段数
    """
    @staticmethod
    def calculate(x: np.ndarray, segment: int = 5)->Tuple[np.ndarray, np.ndarray]:
        """计算分布


        Parameters:
            x (np.ndarray): - 复杂的多层序列
            segment (int): -  均分的分段数,默认100
        Returns:
            Tuple[np.ndarray, np.ndarray]: - 返回用于画图的X,Y
        """
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
        """输出不同分段中数组中的元素每个分段中的出现的个数
        """
        return dict(list(zip(self.xticks, self.Y)))

    def __init__(self, x: np.ndarray, segment: int = 100):
        self.__x = x
        self.segment = segment
        self.calcul_distribution()

    def calcul_distribution(self):
        self.__Y, self.__X = Distribution.calculate(self.x, self.segment)
        self.__xticks = [str(self.X[i]) + '~' + str(self.X[i + 1]) for i in range(len(self.Y))]

    def draw(self):
        """画出分布情况
        """
        xticks = self.xticks
        plt.bar(range(len(self.Y)), self.Y)
        for i, j in zip(range(len(self.Y)), self.Y):
            plt.text(i, j + 0.5, str(float(self.Y[i]) / sum(self.Y) * 100) + "%")

        plt.xlim(-1, self.segment * 1.1)
        plt.xticks(range(len(self.xticks)))
        plt.show()
