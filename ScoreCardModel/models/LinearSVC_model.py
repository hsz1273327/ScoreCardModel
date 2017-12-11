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

    class MySvc(LinearSVC):
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
                t =  str(self.ds[f].transform([x_.get(f)])[0])
                r = self.woes[f].transform([t])[0]
                result.append(r)
            print(np.array(result))
            return np.array(result)
        
        def _predict_proba(self, x):
            print("*********** _predict_proba")
            print(x)
            return self._model.predict_proba(x)


        def _pre_trade_batch_row(self, row, Y, bins):
            d = Discretization(bins)
            d_row = d.transform(row)
            woe = WeightOfEvidence()
            woe.fit(d_row, Y)
            return d, woe, woe.transform(d_row)

        def pre_trade_batch(self, X, Y):
            self.ds = {}
            self.woes = {}
            self.table = {}
            cols = {
                'mobile_operators_juxinli_report__contact_region':[-1,8,10,20,27,100],
                'mobile_operators_juxinli_report__phone_gray_score':[-1,15,42,68,82,100000],
                'mobile_operators_juxinli_report__contacts_class2_blacklist_cnt':[-1,218,654,1000000],
                'mobile_operators_juxinli_report__contacts_class1_cnt':[-1,34,67,100,133,199,1000000],
                'mobile_operators_juxinli_report__contacts_router_cnt':[-1,26,52,78,1000000],
                'basic_info__age':[-1,21,24,27,29,32,35,37,40,43,100],
                'mobile_basic__call_time':[-1,680,1250,2394,2966,999999],
                'addressbook_validate__counts':[-1,577,1152,1000000],
                'call_records_feature__max_place_rate':[-0.1,0.5,0.95,1.0],
                'mobile_basic__sustained_days':[-1,150,174,300]
            }
            for i,bins in cols.items():
                self.ds[i], self.woes[i], self.table[i] = self._pre_trade_batch_row(
                    X[i], Y, bins)
            return pd.DataFrame(self.table)

    model = MySvc()
    model.train(l,z)
    model.predict(l.loc[0].to_dict())
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
        from sklearn.svm import LinearSVC
        model = LinearSVC(**kwargs)
        model.fit(X_matrix, y)
        return model
