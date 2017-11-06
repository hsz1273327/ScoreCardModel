r"""评分卡
===============

用于二分类问题通过模型预测的概率来计算得分


计算公式:
-----------------

.. math:: factor = {\frac {p} {log(2)}}
.. math:: offset = b - p \cdot {\frac {log(o)} {log(2)}}
.. math:: odds = {\frac {p_t} {p_f}}
.. math:: score = factor \cdot {log(odds)} + offset

使用方法:
------------


>>> sc = ScoreCardModel(model)
>>> sc.predict(x)

评分卡类默认会使用包装的分类器的`predict`和`pre_trade`方法,
我们也可以适当的重写评分卡的这两个方法来满足业务要求.

"""
import numpy as np
from .mixins.serialize_mixin import SerializeMixin


class ScoreCardModel(SerializeMixin):
    """
    本模型需要使用一个已经训练好的分类器来初始化,预测,计算评分也都是依赖于它.

    Attributes:

        _model (ScoreCradModel.models.meta): - 训练好的预测模型
        b (int): - 偏置量的常数项,用于作为基数
        o (int): - 用于计算偏置量
        p (int): - 用于计算偏置量和因数项
        round_ (int): - 精度

    """

    def __init__(self, model, b=100, o=1, p=20, round_=1):
        self._model = model
        self.b = b
        self.o = o
        self.p = p
        self.round_ = round_

    def pre_trade(self, x):
        """"数据预处理,预测的时候由于输入未必是处理好的,因此需要先做下预处理"""
        return self._model.pre_trade(x)

    def predict(self, x):
        """
        Parameters:

            x (Sequence): - 用于分段的序列

        Returns:

            float: - 预测出来的分数

        """
        x = self.pre_trade(x)
        proba = self._model._predict_proba(x)
        factor = self.p / np.log(2)
        offset = self.b - self.p * (np.log(self.o) / np.log(2))
        p_f, p_t = proba[0]
        odds = p_t / p_f
        return round(factor * np.log(odds) + offset, self.round_)
