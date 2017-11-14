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


class KS:
    def __init__(self, ks, good_bad_rate, bad_rate, good_rate):
        self.ks = ks
        self.good_bad_rate = good_bad_rate
        self.bad_rate = bad_rate
        self.good_rate = good_rate


class ScoreCardWithKSModel(ScoreCardModel):
    """
    本模型需要使用一个已经训练好的分类器来初始化.预测,计算评分也都是依赖于它.
    它是ScoreCardModel的子类,并且可以用ScoreCardModel或者它的子类实例化.
    这个类主要是新增了一些类方法.之所以弄个子类,是为了序列化的时候不用引入matplotlib作为依赖.
    这个类更加合适离线使用,最好别用来序列化.

    Attributes:

        _model (ScoreCradModel.models.meta): - 训练好的预测模型
        b (int): - 偏置量的常数项,用于作为基数
        o (int): - 用于计算偏置量
        p (int): - 用于计算偏置量和因数项
        round_ (int): - 精度

    """

    @classmethod
    def From_scorecard(clz, obj):
        obj_n = clz(model=obj._model, b=obj.b, o=obj.o,
                    p=obj.p, round_=obj.round_)
        obj_n.pre_trade = obj.pre_trade
        obj_n.predict = obj.predict
        return obj_n

    @classmethod
    def Threshold_to_score(clz, X_score, threshold=0.2):
        """可以通过得分获取其在训练数据总体上的百分位数,并打印以此为阈值时的预测效果.
        也可以通过得分获取其在某一个数据集上的百分位数.

        Attributes:

            X_score (Sequence[number]): - 预测的得分
            threshold (float): - 阈值百分位数,0~1之间,按从大到小的顺序取值,0.1代表从大到小前10%
        """
        score_array = np.array(X_score)
        t = 100 - threshold * 100
        return np.percentile(score_array, t)

    @classmethod
    def Score_to_threshold(clz, X_score, *, y=None, score=100, round_=4):
        """可以通过得分获取其在训练数据总体上的百分位数,并打印以此为阈值时的预测效果.
        也可以通过得分获取其在某一个数据集上的百分位数.

        Attributes:

            X_score (Sequence[number]): - 预测的得分
            y (Sequence[number]): - 实际的标签,可以为空
            score (number): - 阈值分,大于它的为T,否则为F
        """

        score_array = np.array(X_score)
        if y is not None:
            print(precision_score(y, (score_array > score), average='macro'))
            print(classification_report(y, (score_array > score)))
        return round(len(score_array[score_array > score]) / len(score_array),
                     round_)

    @classmethod
    def Get_ks(clz, X_score, y, *, threshold=0.1, threshold_score=None):
        """计算在某处时的ks值,可以是阈值百分位数,也可以是某个分数

        Attributes:

            X_score (Sequence[number]): - 预测的得分
            y (Sequence[number]): - 实际的标签
            threshold (float): - 阈值百分位数,默认为0.1也就是前10%
            threshold_score (number): - 位置阈值分,取在它所在排序位置之前(包含这个分数的位置)的计算ks值
            score (number): - 阈值分,大于等于它的是True,否贼是False
        """
        org_data = list(zip(X_score, y))
        good_total = len(y[y == 1])
        bad_total = len(y[y != 1])
        ordered_data = sorted(org_data, key=lambda x: x[0], reverse=True)
        if not threshold_score:
            threshold_score = clz.Threshold_to_score(X_score, threshold)
        filted_data_gen = (i for i in ordered_data if i[0] >= threshold_score)
        target_y = [j for _, j in filted_data_gen]

        def _get_ks(y_true):
            good = len(y_true[y_true == 1])
            bad = len(y_true[y_true != 1])
            bad_rate = bad / bad_total
            good_rate = good / good_total
            ks = abs(good_rate - bad_rate)
            if (good + bad) == 0:
                good_bad_rate = 0
            else:
                good_bad_rate = good / (good + bad)
            return KS(ks, good_bad_rate, bad_rate, good_rate)
        ks = _get_ks(np.array(target_y))
        return ks

    @classmethod
    def Drawks(clz, X_score, y):
        """绘制ks曲线

        Attributes:

            X_score (Sequence[number]): - 预测的得分
            y (Sequence[number]): - 实际的标签
        """
        import matplotlib.pyplot as plt
        X = []
        Y_ks = []
        for i in range(100):
            x = i / 100
            ks = clz.Get_ks(X_score, y, threshold=i / 100)
            X.append(x)
            Y_ks.append(ks)

        plt.plot(X, [i.good_rate for i in Y_ks], color="blue", label="good rat")
        plt.plot(X, [i.bad_rate for i in Y_ks], color="red", label="bad rat")
        plt.plot(X, [i.ks for i in Y_ks], color="yellow", label="KS")
        plt.plot(X, [i.good_bad_rate for i in Y_ks],
                 color="green", label="good_bad")
        plt.legend(loc='upper center'),
        plt.show()

    def _calcul_X_score(self, X):
        X_score = []
        for i in range(len(X)):
            score = self.predict(self.pre_trade(X[i]))
            X_score.append(score)
        return X_score

    def threshold_to_score(self, X, threshold=0.2):
        """可以通过得分获取其在训练数据总体上的百分位数,并打印以此为阈值时的预测效果.
        也可以通过得分获取其在某一个数据集上的百分位数.

        Attributes:

            X (pd.DateFrame): - 待预测特征数据
            threshold (float): - 阈值百分位数,0~1之间,按从大到小的顺序取值,0.1代表从大到小前10%
        """
        X_score = self._calcul_X_score(X)
        return self.__class__.Threshold_to_score(X_score, threshold=threshold)

    def score_to_threshold(self, X, *, y=None, score=100, round_=4):
        """可以通过得分获取其在训练数据总体上的百分位数,并打印以此为阈值时的预测效果.
        也可以通过得分获取其在某一个数据集上的百分位数.

        Attributes:

            X (pd.DateFrame): - 待预测特征数据
            y (Sequence[number]): - 实际的标签,可以为空
            score (number): - 阈值分,大于它的为T,否则为F
        """
        X_score = self._calcul_X_score(X)
        return self.__class__.Score_to_threshold(X_score, y=y, score=score, round_=round_)

    def get_ks(self, X, y, *, threshold=0.1, threshold_score=None):
        """计算在某处时的ks值,可以是阈值百分位数,也可以是某个分数

        Attributes:

            X (pd.DateFrame): - 待预测特征数据
            y (Sequence[number]): - 实际的标签
            threshold (float): - 阈值百分位数,默认为0.1也就是前10%
            threshold_score (number): - 位置阈值分,取在它所在排序位置之前(包含这个分数的位置)的计算ks值
            score (number): - 阈值分,大于等于它的是True,否贼是False
        """
        X_score = self._calcul_X_score(X)
        return self.__class__.Get_ks(X_score, y=y, threshold=threshold, threshold_score=threshold_score)

    def drawks(self, X, y):
        """绘制ks曲线

        Attributes:

            X (pd.DateFrame): - 待预测特征数据
            y (Sequence[number]): - 实际的标签

        """
        X_score = self._calcul_X_score(X)
        self.__class__.Drawks(X_score, y)
