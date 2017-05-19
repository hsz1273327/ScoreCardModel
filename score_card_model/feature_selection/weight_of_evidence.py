import numpy as np
from typing import Any,Union,Sequence, Tuple, Dict

from score_card_model.utils.check import check_array_binary
from score_card_model.utils.count import count_binary
class Woe:
    def __call__(self, X:np.ndarray,
            y:np.ndarray,
            event:Any=1,N:Union[int,Sequence[int]]=5)->Tuple[np.ndarray,np.ndarray]:
        return self.run(X=X,y=y,event=event,N=N)

    def __init__(self):
        self._WOE_MIN = -20
        self._WOE_MAX = 20

    def run(self, X:np.ndarray,
            y:np.ndarray,
            event:Any=1,N:Union[int,Sequence[int]]=5)->Tuple[np.ndarray,np.ndarray]:
        '''
        Calculate woe of each feature category and information value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable which should be binary
        :param event: value of binary stands for the event to predict
        :return: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
                 numpy array of information value of each feature
        '''
        self.check_target_binary(y)
        X1 = self.feature_discretion(X)

        res_woe = []
        res_iv = []
        for i in range(0, X1.shape[-1]):
            x = X1[:, i]
            woe_dict, iv1 = self.woe_single_x(x, y, event)
            res_woe.append(woe_dict)
            res_iv.append(iv1)
        return np.array(res_woe), np.array(res_iv)

    def _woe_single_x(self, x:np.ndarray, tag:np.ndarray, event:Any=1)->Tuple[Dict[int,float],float]:
        '''
        对单独一项自变量(列,特征)计算其woe和iv值.
        woe计算公式:

        :math: `woe_i = log(\frac {\frac {Bad_i} {Bad_{total}}} {\frac {Good_i} {Good_{total}}})`


        :param x: 1-D numpy starnds for single feature
        :param y: 1-D numpy array target variable
        :param event: value of binary stands for the event to predict
        :return: dictionary contains woe values for categories of this feature
                 information value of this feature
        '''
        if not check_array_binary(tag):
            raise AttributeError("tag must be a binary array")

        event_total, non_event_total = count_binary(y, event=event)
        x_labels = np.unique(x)
        woe_dict = {}
        iv = 0
        for x1 in x_labels:
            y1 = y[np.where(x == x1)[0]]
            event_count, non_event_count = count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            if rate_event == 0:
                woe1 = self._WOE_MIN
            elif rate_non_event == 0:
                woe1 = self._WOE_MAX
            else:
                woe1 = math.log(rate_event / rate_non_event)
            woe_dict[x1] = woe1
            iv += (rate_event - rate_non_event) * woe1
        return woe_dict, iv

    def woe_replace(self, X, woe_arr):
        '''
        replace the explanatory feature categories with its woe value
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param woe_arr: numpy array of woe dictionaries, each dictionary contains woe values for categories of each feature
        :return: the new numpy array in which woe values filled
        '''
        if X.shape[-1] != woe_arr.shape[-1]:
            raise ValueError('WOE dict array length must be equal with features length')

        res = np.copy(X).astype(float)
        idx = 0
        for woe_dict in woe_arr:
            for k in woe_dict.keys():
                woe = woe_dict[k]
                res[:, idx][np.where(res[:, idx] == k)[0]] = woe * 1.0
            idx += 1

        return res

    def combined_iv(self, X, y, masks, event=1):
        '''
        calcute the information vlaue of combination features
        :param X: 2-D numpy array explanatory features which should be discreted already
        :param y: 1-D numpy array target variable
        :param masks: 1-D numpy array of masks stands for which features are included in combination,
                      e.g. np.array([0,0,1,1,1,0,0,0,0,0,1]), the length should be same as features length
        :param event: value of binary stands for the event to predict
        :return: woe dictionary and information value of combined features
        '''
        if masks.shape[-1] != X.shape[-1]:
            raise ValueError('Masks array length must be equal with features length')

        x = X[:, np.where(masks == 1)[0]]
        tmp = []
        for i in range(x.shape[0]):
            tmp.append(self.combine(x[i, :]))

        dumy = np.array(tmp)
        # dumy_labels = np.unique(dumy)
        woe, iv = self.woe_single_x(dumy, y, event)
        return woe, iv

    def combine(self, list):
        res = ''
        for item in list:
            res += str(item)
        return res
