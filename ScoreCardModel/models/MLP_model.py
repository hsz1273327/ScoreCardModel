"""多层感知器模型(神经网络)
=========================

最简单基础的神经网络分类器模型,神经网络当然不光可以用来做二分类,但是评分卡要的就是二分类.


用法
---------------------

由于数据进来格式千奇百怪,这个模型最好的用法是继承后重写
`predict`,`pre_trade`,`pre_trade_batch`这几个方法,适当的也可以重写`train`方法.

不过一般来说神经网络不要求分箱.


.. code:: python

    class MyMlp(MLPModel):
    
        def predict(self, x):
            x = self.pre_trade(x)
            print(x)
            return self._predict_proba(x)

        def pre_trade(self, x):
            x_ ={k+"__"+l:u for k,v in x.items() for l,u in v.items()}
            print("***********x_")
            import numpy as np
            result = []
            for f in self.feature_order:
                for k,v in x_.items():
                    if k == i:
                        result.append(v)
                        break
            y = self.sca.transform(result)
            return y
        
        def _predict_proba(self, x):
            print("*********** _predict_proba")
            print(x)
            return self._model.predict_proba(x)



        def pre_trade_batch(self, X, Y):
            from sklearn import preprocessing
            sca = preprocessing.StandardScaler()
            sca.fit(X)
            self.sca = sca
            X_train = sca.transform(X)
            return X_train
        
    lr = MyMlp()
    lr.train(l,z)
    lr.predict(l.loc[0].to_dict())
"""

from .meta import Model
from ..mixins.serialize_mixin import SerializeMixin


class MLPModel(Model, SerializeMixin):
    """该类最好是继承了使用,继承后重写`predict`和`pre_trade`

    Attributes:

         feature_order (Sequence): - 特征顺序
         _model (sklearn.model): - 训练出来的sklearn的分类器模型


    """
    feature_order = None
    _model = None

    def predict(self, x):
        """预测用的接口,根据需求重写实现"""
        return self._predict(x)

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
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier(**kwargs)
        model.fit(X_matrix, y)
        return model
