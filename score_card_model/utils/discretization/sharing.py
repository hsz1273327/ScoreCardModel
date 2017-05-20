"""
百分位数均分离散化
"""
from typing import Union, Sequence, Tuple, List
from collections import Sequence as seq
import numpy as np
from scipy import stats


def get_sections(x: np.ndarray, n: int=5)->List[Tuple[int, float, float]]:

    if not isinstance(n, int) or n <= 0:
        raise AttributeError("n must be a positive int")
    step = 100 / n
    result = []
    for i in range(n):
        point1 = stats.scoreatpercentile(x, i * step)
        if i == (n - 1):
            point2 = stats.scoreatpercentile(x, 100)
        else:
            point2 = stats.scoreatpercentile(x, (i + 1) * step)
        result.append((i, point1, point2))
    return result


def discrete(x: np.ndarray, n: int=5)->np.ndarray:
    '''
    计算出每位数据所处的百分位数区间位置,默认分为5段,段数从1开始计数,数值越大越接近100%
    '''
    if (isinstance(n, int) and n <= 0) or isinstance(x[0], np.str_):
        return x
    else:
        res = np.array([0] * x.shape[-1], dtype=int)
        points = get_sections(x=x, n=n)
        for i, point1, point2 in points:
            x1 = x[np.where((x >= point1) & (x <= point2))]
            mask = np.in1d(x, x1)
            res[mask] = (i + 1)
        return res


def discrete_features(X: np.ndarray, N: Union[int, Sequence[int]]= 5)->np.ndarray:
    '''
    计算特征矩阵每一项特征的百分位数均分离散化,
    参数X必须为二维数组,且特征为列,数据为行,
    如果N中的某一项为0,那么该项不分区间认为是离散的
    '''
    X = X.T
    if isinstance(N, str) or not isinstance(N, seq):
        N = [N for i in range(X.shape[0])]
    if len(N) != X.shape[0]:
        raise AttributeError("N must have the same len with the rows' len")
    temp = []
    for i in range(X.shape[0]):
        x = X[i, :]
        x1 = discrete(x, n=N[i])
        temp.append(x1)
    return np.array(temp).T
