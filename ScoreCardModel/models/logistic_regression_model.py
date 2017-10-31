"""logistic回归模型
=========================




用法
---------------------


:: python


class MyLR(LogisticRegressionModel):


"""

from .meta import Model
from ..mixins.serialize_mixin import SerializeMixin


class LogisticRegressionModel(Model, SerializeMixin):
    """该类最好是继承了使用,继承后重写`predict`和`pre_trade`

    Attributes:

         feature_order (Sequence): - 特征顺序
         _model (sklearn.model): - 训练出来的sklearn的分类器模型


    """
    feature_order = None
    _model = None

    def _predict(self, x):
        """二分类预测

        Parameters:

            x (Sequence): - 用于预测的特征向量

        Returns:

            bool: - 返回0,1也就是False/True,True表示预测值为True,否则说明预测值为False


        """
        result = self.model.predict(x)
        return result[0]

    def predict(self, x):
        """预测用的接口,根据需求重写实现"""
        return self._predict(x)

    def _predict_proba(self, x):
        """
        不同预测值的概率

        Parameters:

            x (Sequence): - 用于预测的特征向量

        Returns:

            float: - 预测值为False的概率
            float: - 预测值为True的概率


        """
        return self.model.predict_proba(x)

    def pre_trade(self, x):
        """"数据预处理,预测的时候由于输入未必是处理好的,因此需要先做下预处理"""
        import numpy as np
        y_ = {k + "_" + l: u for k, v in x.items() for l, u in v.items()}
        y__ = []
        for i in self.feature_order:
            for k, v in y_.items():
                if k == i:
                    y__.append(v)
                    break

        x = [np.array(y__)]
        return x

    def pre_trade_batch(self, X):
        """批量数据预处理,预测的时候由于输入未必是处理好的,因此需要先做下预处理"""
        result = []
        for x in X:
            result += self.pre_trade(x)
        return result

    def _train(self, X_matrix, y, **kwargs):
        """训练数据

        Parameters:

            X_matrix (numpy.array): - 由训练数据组成的特征矩阵
            y (numpy.array): - 特征数据对应的标签向量

        Returns:

            sklearn.model: - sklearn的模型

            
        """
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(**kwargs)
        lr.fit(X_matrix, y)
        return lr

