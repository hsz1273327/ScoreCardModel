"""
评分卡模型
==============

所谓评分卡其实只是一种用于表现预测概率的方式,书上说是用于给用户信用打分的.
通常它会结合logistic回归使用.当然了,个人认为用其他分类器也是一样的.

连续特征往往是非线性的,因此预测模型有两种思路

* 先做一些特征工程,通常工业上会将连续数据分段,把它转化为一组离散数据.
    如何分段也就是这种思路下预测模型准确度的根本了.

* 使用深度学习,深度学习自带特征抽象能力.

本模型就是第一种思路的实践.



模块设计:
----------------

本模块有如下几个子模块:

* 连续数据离散化模块 discretization
* 计算离散数据各枚举值的证据权重 weight_of_evidence
* 分类器模型 models,其中现在只有logistic回归一个实现.
* 用于计算得分的评分卡模型 score_card


依赖:
--------
本模块依赖numpy,sklearn,pandas


使用方法:
----------
任何机器学习模型,使用流程都差不多

* 清洗特征,本模型不管
* 观察数据,使用`Discretization`来将连续数据离散化. 
* 将离散化后的数据每一列都使用`WeightOfEvidence`来计算离散数据的证据权重,每一项计算出来都可以计算出一个iv值(信息值),都算得了可以比较了挑大的选出来使用.
* 将离散数据都用`WeightOfEvidence`计算出每项的权重替换这些分类,然后拿这个放入分类器训练
* 重复上面的步骤找到效果最好的一个模型.
* 初始化一个`ScoreCardModel`用于计算最终得分


一个完整的例子:
------------------

>>> from sklearn import datasets
>>> import pandas as pd
>>> from ScoreCardModel.discretization import Discretization
>>> from ScoreCardModel.weight_of_evidence import WeightOfEvidence
>>> from ScoreCardModel.models.logistic_regression_model import LogisticRegressionModel
>>> from ScoreCardModel.binning.score_card import ScoreCardModel
>>>
>>> class MyLR(LogisticRegressionModel):
>>>     def predict(self, x):
>>>          x = self.pre_trade(x)
>>>          return self._predict_proba(x)
>>>      
>>>     def pre_trade(self, x):
>>>         import numpy as np
>>>         result = []
>>>         for i,v in x.items():
>>>             t = self.ds[i].transform([v])[0]
>>>             r = self.woes[i].transform([t])[0]
>>>             result.append(r)
>>>         return np.array(result)
>>>
>>>     def _pre_trade_batch_row(self,row,Y,bins):
>>>         d = Discretization(bins)
>>>         d_row = d.transform(row)
>>>         woe = WeightOfEvidence()
>>>         woe.fit(d_row,Y)
>>>         return d,woe,woe.transform(d_row)
>>>     
>>>     def pre_trade_batch(self, X,Y):
>>>         self.ds = {}
>>>         self.woes = {}
>>>         self.table = {}
>>>         self.ds["sepal length (cm)"],self.woes["sepal length (cm)"],self.table["sepal length (cm)"]= self._pre_trade_batch_row(
>>>             X["sepal length (cm)"],Y,[0,2,5,8])
>>>         self.ds['sepal width (cm)'],self.woes['sepal width (cm)'],self.table['sepal width (cm)'] = self._pre_trade_batch_row(
>>>             X['sepal width (cm)'],Y,[0,2,2.5,3,3.5,5])
>>>         self.ds['petal length (cm)'],self.woes['petal length (cm)'],self.table['petal length (cm)'] = self._pre_trade_batch_row(
>>>             X['petal length (cm)'],Y,[0,1,2,3,4,5,7])
>>>         self.ds['petal width (cm)'],self.woes['petal width (cm)'],self.table['petal width (cm)'] = self._pre_trade_batch_row(
>>>             X['petal width (cm)'],Y,[0,1,2,3])
>>>         return pd.DataFrame(self.table)
>>>
>>> iris = datasets.load_iris()
>>> y = iris.target
>>> z = (y==0)
>>> l = pd.DataFrame(iris.data,columns=iris.feature_names)
>>> lr = MyLR()
>>> lr.train(l,z)
>>> lr.predict(l.loc[0].to_dict())
array([[ 0.46315882,  0.53684118]])
>>> sc = ScoreCardModel(lr)
>>> sc.predict(l.loc[0].to_dict())
104.3



模型序列化
------------

要实际应用模型我们就得想办法保存我们的训练成果,也就是序列化.这个包使用dill做python的序列化主要工具,
而为了用于传递和存入数据库,我们又使用base64再编码一次.

无论是分类模型还是评分卡模型,都混入了`SerializeMixin`,这个Mixin提供了对象的序列化能力
和从这个序列化后的字符串转化到模型对象的能力.其接口为`obj.dumps`和`clz.loads`

示例:

>>> sc_str = sc.dumps()
>>> sc_l = ScoreCardModel.loads(sc_str)
>>> sc_l.predict(l.loc[0].to_dict())
104.3


"""
