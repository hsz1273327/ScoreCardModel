"""线性支持向量分类器
=========================

与参数kernel ='linear'的SVC类似，但是用liblinear而不是libsvm来实现，
所以它在惩罚和丢失函数的选择上有更大的灵活性，并且应该更好地适应大量的样本.

这个类同时支持密集和稀疏输入，并且多类支持是按照`one-vs-the-rest`的方式处理的.


用法
---------------------

由于数据进来格式千奇百怪,这个模型最好的用法是继承后重写
`predict`,`pre_trade`,`pre_trade_batch`这几个方法,适当的也可以重写`train`方法.

不过一般来说神经网络不要求分箱.


.. code:: python

    class MyLR(LogisticRegressionModel):
        def predict(self, x):
            x = self.pre_trade(x)
            return self._predict_proba(x)

        def pre_trade(self, x):
            import numpy as np
            result = []
            for i, v in x.items():
                t = self.ds[i].transform([v])[0]
                r = self.woes[i].transform([t])[0]
                result.append(r)
            return np.array(result)
        def _pre_trade_batch_row(self,row,Y,bins):
            d = Discretization(bins)
            d_row = d.transform(row)
            woe = WeightOfEvidence()
            woe.fit(d_row,Y)
            return d,woe,woe.transform(d_row)
        
        def pre_trade_batch(self, X,Y):
            self.ds = {}
            self.woes = {}
            self.table = {}
            self.ds["sepal length (cm)"],self.woes["sepal length (cm)"],self.table["sepal length (cm)"]= self._pre_trade_batch_row(
                X["sepal length (cm)"],Y,[0,2,5,8])
            self.ds['sepal width (cm)'],self.woes['sepal width (cm)'],self.table['sepal width (cm)'] = self._pre_trade_batch_row(
                X['sepal width (cm)'],Y,[0,2,2.5,3,3.5,5])
            self.ds['petal length (cm)'],self.woes['petal length (cm)'],self.table['petal length (cm)'] = self._pre_trade_batch_row(
                X['petal length (cm)'],Y,[0,1,2,3,4,5,7])
            self.ds['petal width (cm)'],self.woes['petal width (cm)'],self.table['petal width (cm)'] = self._pre_trade_batch_row(
                X['petal width (cm)'],Y,[0,1,2,3])
            return pd.DataFrame(self.table)

    lr = MyLR()
    lr.train(l,z)
    lr.predict(l.loc[0].to_dict())
"""

from .meta import Model
from ..mixins.serialize_mixin import SerializeMixin


class LinearSVC(Model, SerializeMixin):
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
        result = self._model.predict(x)
        return result

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
        return self._model.predict_proba(x)

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
        """批量数据预处理,训练的时候由于输入未必是处理好的,因此需要先做下预处理"""
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
        from sklearn.svm import LinearSVC
        model = LinearSVC(**kwargs)
        model.fit(X_matrix, y)
        return lr
