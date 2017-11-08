"""
离散化连续数据
===============

我们使用pandas来对连续数据做离散化处理.其接口尽量与sklearn的数据预处理模块接近.


使用方法:
----------

>>> d = Discretization([0,2,4])
>>> list(d.transform([0.5,1.5,2.5,3.5]))
['(0, 2]', '(0, 2]', '(2, 4]', '(2, 4]']


"""

import numpy as np
import pandas as pd


class Discretization:
    """离散化连续数据.需要实例化以保存bins状态.

    Attributes:

         bins (Sequence): - 用于分段的列表,第一位为下限,最后一位为上限

"""

    def __init__(self, bins):
        self.bins = bins

    def transform(self, x):
        """
        Parameters:

            x (Sequence): - 用于分段的序列

        Returns:

            np.array: - 以分段字符串为枚举内容的numpy序列

        """
        s = pd.cut(x, bins=self.bins)
        d = pd.get_dummies(s)
        z = d.T.to_dict()
        re = []
        for i, v in z.items():
            for j, u in v.items():
                if u == 1:
                    re.append(str(j))
        return np.array(re)
