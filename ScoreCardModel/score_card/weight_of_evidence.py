r"""计算Woe
==============

针对离散数据,我们需要将离散的枚举值替换成数值才可以用于计算.这些数值就是各个枚举值的权重.
广义上讲,woe可以算是一种编码方式.
如何计算这些权重呢?这就得训练.我们需要一组二值代表签数据,
通过统计离散特征不同枚举值对目标数据的响应情况来计算触发概率,


触发概率
---------

本模型针对二分类问题,事件也就是只有True,False两种.我们当然认为True,枚举值i的触发概率可以这样计算,
比如某个枚举值i对true的触发概率,就是所有i值时是true的数量除以总的t的数量

.. math:: p_f = {\frac {f_i} {f_{total}}}

.. math:: p_t = {\frac {t_i} {t_{total}}}


woe
------

woe就是计算正负概率的信息值

.. math:: woe_i = log_2({\frac {p_f} {p_t}})


iv
-----

iv值就是这以特征的总信息量也就是各枚举值信息量的和

.. math:: IV_i = (p_f-p_t)*log_2({\frac {p_f} {p_t}})

.. math:: IV = \sum_{k=0}^n IV_i


使用方法:
----------

>>> from sklearn import datasets
>>> import pandas as pd
>>> iris = datasets.load_iris()
>>> y = iris.target
>>> z = (y==0)
>>> l = pd.DataFrame(iris.data,columns=iris.feature_names)
>>> d = Discretization([0,5.0,5.5,7])
>>> re = d.transform(l["sepal length (cm)"])
>>> woe = WeightOfEvidence()
>>> woe.fit(re,z)
>>> woe.woe
{'(0, 5]': 2.6390573296152589,
 '(5, 5.5]': 1.5581446180465499,
 '(5.5, 7]': -2.5389738710582761}
>>> woe.iv
3.617034906554693
>>> test = ['(0, 5]', '(0, 5]', '(5.5, 7]', '(5.5, 7]', '(5, 5.5]', '(5, 5.5]',
       '(5.5, 7]', '(5, 5.5]', '(5, 5.5]', '(5, 5.5]', '(0, 5]']
>>> woe.transform(test)
array([ 2.63905733,  2.63905733, -2.53897387, -2.53897387,  1.55814462,
        1.55814462, -2.53897387,  1.55814462,  1.55814462,  1.55814462,
        2.63905733])

"""
import numpy as np
from sklearn.utils.multiclass import type_of_target


class WeightOfEvidence:
    """计算某一离散特征的woe值
    Attributes:

        woe (Dict): - 训练好的证据权重
        iv (Float): - 训练的离散特征的信息量

    """

    def __init__(self):
        self.woe = None
        self.iv = None

    def _posibility(self, x, tag, event=1):
        """计算触发概率

        Parameters:

            x (Sequence): - 离散特征序列
            tag (Sequence): - 用于训练的标签序列
            event (any): - True指代的触发事件

        Returns:

            Dict[str,Tuple[rate_T, rate_F]]: - 训练好后的好坏触发概率

        """
        if type_of_target(tag) not in ['binary']:
            raise AttributeError("tag must be a binary array")
        if type_of_target(x) in ['continuous']:
            raise AttributeError("input array must not continuous")
        tag = np.array(tag)
        x = np.array(x)
        event_total = (tag == event).sum()
        non_event_total = tag.shape[-1] - event_total
        x_labels = np.unique(x)
        pos_dic = {}
        for x1 in x_labels:
            y1 = tag[np.where(x == x1)[0]]
            event_count = (y1 == event).sum()
            non_event_count = y1.shape[-1] - event_count
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            pos_dic[x1] = (rate_event, rate_non_event)
        return pos_dic

    def fit(self, x, y, *, event=1, woe_min=-20, woe_max=20):
        """训练对单独一项自变量(列,特征)的woe值.

        Parameters:

            x (Sequence): - 离散特征序列
            y (Sequence): - 用于训练的标签序列
            event (any): - True指代的触发事件
            woe_min (munber): - woe的最小值,默认值为-20
            woe_max (munber): - woe的最大值,默认值为20

        """
        woe_dict = {}
        iv = 0
        pos_dic = self._posibility(x=x, tag=y, event=event)
        for l, (rate_event, rate_non_event) in pos_dic.items():
            if rate_event == 0:
                woe1 = woe_min
            elif rate_non_event == 0:
                woe1 = woe_max
            else:
                woe1 = np.log(rate_event / rate_non_event)  # np.log就是ln
            iv += (rate_event - rate_non_event) * woe1
            woe_dict[str(l)] = woe1
        self.woe = woe_dict
        self.iv = iv

    def transform(self, X):
        """将离散特征序列转换为woe值组成的序列

        Parameters:

            X (Sequence): - 离散特征序列

        Returns:

            numpy.array: - 替换特征序列枚举值为woe对应数值后的序列

        """
        return np.array([self.woe.get(i) for i in X])
