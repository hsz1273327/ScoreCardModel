
ScoreCardModel
===============================

* version: 1.1.3

* status: dev

* author: hsz

* email: hsz1273327@gmail.com

Desc
--------------------------------

a simple tool for score card model


keywords:math,finance


Feature
----------------------

* Serializable
* mutil classifier model support
* ks-curve support

Change
-------------------------

* scorecard now can set a threshold value to return a bool result

Example
-------------------------------

>>> from sklearn import datasets
>>> import pandas as pd
>>> from ScoreCardModel.binning.discretization import Discretization
>>> from ScoreCardModel.weight_of_evidence import WeightOfEvidence
>>> from ScoreCardModel.models.logistic_regression_model import LogisticRegressionModel
>>> from ScoreCardModel.score_card import ScoreCardModel
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
>>> sc.predict(sc.pre_trade(l.loc[0].to_dict()))
104.3
>>> scs = []
>>> for i in range(len(l)):
>>>    score = sc.predict(sc.pre_trade(l.loc[i].to_dict()))
>>>    scs.append(score)
>>> print(ScoreCardWithKSModel.Threshold_to_score(scs, 0.5))
1.0
>>> print(ScoreCardWithKSModel.Score_to_threshold(scs, score=70))
1.0
.
.
.
>>> print(ScoreCardWithKSModel.Score_to_threshold(scs, y=z, score=100))
0.3467
>>> print(ScoreCardWithKSModel.Get_ks(scs, y=z, threshold=0.4).ks)
0.9
>>> # ScoreCardWithKSModel.Drawks(scs, y=z)
>>> scsc = [l.loc[i].to_dict() for i in range(len(l))]
>>> scks = ScoreCardWithKSModel.From_scorecard(sc)
>>> print(scks.threshold_to_score(scsc, 0.5))
1.0
>>> print(scks.score_to_threshold(scsc, score=70))
1.0
.
.
.
>>> print(scks.score_to_threshold(scsc, y=z, score=100))
0.3467
>>> print(scks.get_ks(scsc, y=z, threshold=0.4).ks)
0.9
>>> scks.drawks(scsc, y=z)


Install
--------------------------------

- ``python -m pip install ScoreCardModel``


Documentation
--------------------------------

`Documentation on Readthedocs <https://data-science-tools.github.io/ScoreCardModel/>`_.





