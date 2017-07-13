#coding:utf-8
import numpy as np
from sklearn.linear_model import LogisticRegression
from ..utils.discretization.split import split
from score_card_model.feature_selection.weight_of_evidence import Woe
class Core(object):
    def __init__(self,X, tag,
                 discrete,feature_order,
                 event=1,
                 label=None,
                 model_type = LogisticRegression,
                 min_v=-20, max_v=20):

        self.X = X
        self.tag = tag
        self.event =event
        self.discrete = discrete
        self.feature_order = feature_order
        self.label = label
        self._model_type = model_type
        self.min_v = min_v
        self.max_v = max_v

        self.N = X.shape[-1]
        self._discrete =[(split,{"n":i}) for i in self.discrete]
        woe = Woe(X=self.X, tag=self.tag, event=self.event,label=self.label,
                  discrete=self._discrete,min_v=self.min_v, max_v=self.max_v)
        self.woe = woe.woe
        self.X_label = woe.X_result
        woe_X = []
        self.iv = woe.iv
        if self.label:
            for i in range(self.X.shape[-1]):
                array = self.X_label[:, i]
                woe_i = woe.get(self.label[i])
                line = np.array([woe_i.get(i) for i in array])
                woe_X.append(line)

        else:
            for i in range(self.X.shape[-1]):
                array = self.X_label[:, i]
                woe_i = self.woe.get(str(i))
                line = np.array([woe_i.get(i) for i in array])
                woe_X.append(line)

        self.woe_X = np.array(woe_X).T
