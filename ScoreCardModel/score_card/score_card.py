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
>>> sc.predict(sc.pre_trade(x))

评分卡类默认会使用包装的分类器的`predict`和`pre_trade`方法,
我们也可以适当的重写评分卡的这两个方法来满足业务要求.


KS 曲线:
--------------

不知为何,搞经济金融的喜欢用KS曲线来评估评分卡的效果.abs

所谓KS曲线计算方法很简单:

1. 将得分与实际标签合并后以得分从大到小排序,这个序列设为total

2. 计算出总共标签中的好标签数量和坏标签数量good_total,bad_total

3. 获取total前i%的用户计算其中好用户数量good和坏用户数量bad并计算(good/good_total-bad/bad_total)的绝对值,这个值就是i%位的ks值,i从0计算到100,这样得到i%和每位对应的ks值就可以用于绘制x轴和y轴.

4. 将这两个序列画出来也就得到了ks曲线图.

在这个计算过程中另外有意义的几个值为:

好坏比 good/bad

好占比 good/good_total

坏占比 bad/bad_total

"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import classification_report, precision_score
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
        threshold (float): - 阈值,可选

    """

    def __init__(self, model, b=100, o=1, p=20, round_=1,threshold=None):
        self._model = model
        self.b = b
        self.o = o
        self.p = p
        self.threshold = threshold
        self.round_ = round_

    def pre_trade(self, x):
        """"数据预处理,预测的时候由于输入未必是处理好的,因此需要先做下预处理"""
        return self._model.pre_trade(x)

    def predict(self, x):
        """用于预测某一条预处理过的特征向量得分的方法

        Parameters:

            x (Sequence): - 用于分段的序列

        Returns:

            float: - 预测出来的分数
            bool: - 预测的分数超过阈值则返回True,否则False

        """
        proba = self._model._predict_proba([x])
        factor = self.p / np.log(2)
        offset = self.b - self.p * (np.log(self.o) / np.log(2))
        p_f, p_t = proba[0]
        odds = p_t / p_f
        score = round(factor * np.log(odds) + offset, self.round_)
        if self.threshold:
            return True if score > self.threshold else False
        else:
            return score

