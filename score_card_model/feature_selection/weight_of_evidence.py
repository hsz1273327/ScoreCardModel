import numpy as np
from collections import OrderedDict
from typing import Any, Union, Sequence, Tuple, Dict, Callable, List
from score_card_model.utils.discretization.sharing import discrete
from score_card_model.utils.check import check_array_binary, check_array_continuous
from score_card_model.utils.count import count_binary


class Woe:
    """
    注意woe和iv方法如果有discrete参数,必须是tuple(func,dict)或者list(tuple(func,dict))
    """
    WOE_MIN = -20
    WOE_MAX = 20

    @staticmethod
    def _posibility(x: np.ndarray, tag: np.ndarray, event: Any=1, discrete: Callable = None, **kwargs)->Dict[any, Tuple[float, float]]:
        """
        计算占总体的好坏占比
        """
        if discrete:
            x = discrete(x, **kwargs)
        if not check_array_binary(tag):
            raise AttributeError("tag must be a binary array")
        if check_array_continuous(x):
            raise AttributeError("input array must not continuous")
        event_total, non_event_total = count_binary(tag, event=event)
        x_labels = np.unique(x)
        pos_dic = {}
        for x1 in x_labels:
            y1 = tag[np.where(x == x1)[0]]
            event_count, non_event_count = count_binary(y1, event=event)
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            pos_dic[x1] = (rate_event, rate_non_event)
        return pos_dic

    @staticmethod
    def weight_of_evidence(x: np.ndarray, tag: np.ndarray, event: Any=1, woe_min=-20, woe_max=20, discrete: Callable = None, **kwargs)->Dict[any, float]:
        '''
        对单独一项自变量(列,特征)计算其woe和iv值.
        woe计算公式:
        :math: `woe_i = log(\frac {\frac {Bad_i} {Bad_{total}}} {\frac {Good_i} {Good_{total}}})`
        '''
        woe_dict = {}
        pos_dic = Woe._posibility(x=x, tag=tag, event=event, discrete=discrete, **kwargs)
        for l, (rate_event, rate_non_event) in pos_dic.items():
            if rate_event == 0:
                woe1 = woe_min
            elif rate_non_event == 0:
                woe1 = woe_max
            else:
                woe1 = np.log(rate_event / rate_non_event)  # np.log就是ln
            woe_dict[l] = woe1
        return woe_dict

    @staticmethod
    def information_value(x: np.ndarray, tag: np.ndarray, event: Any=1, woe_min=-20, woe_max=20, discrete: Callable = None, **kwargs)->float:
        iv = 0
        pos_dic = Woe._posibility(x=x, tag=tag, event=event, discrete=discrete, **kwargs)
        for l, (rate_event, rate_non_event) in pos_dic.items():
            if rate_event == 0:
                woe1 = woe_min
            elif rate_non_event == 0:
                woe1 = woe_max
            else:
                woe1 = np.log(rate_event / rate_non_event)
            iv += (rate_event - rate_non_event) * woe1
        return iv

    def __call__(self, X: np.ndarray, tag: np.ndarray, event: Any=1, label=None,
                 discrete: Union[None, Tuple[Callable, Dict], Sequence[Tuple[Callable, Dict]]] = None)->Dict[Any, Dict[Any, float]]:
        if label:
            len(label) != X.shape[-1]
            raise AttributeError("label must have the same len with the features' number")
        result = {}
        if discrete:
            if isinstance(discrete, tuple):
                discrete = [discrete for i in range(X.shape[-1])]
            if not label:
                for i in range(X.shape[-1]):
                    func, kws = discrete[i]
                    result[str(i)] = {'woe': self.woe_single_x(X[:, i], tag, event, discrete=func, **kws),
                                      'iv': self.iv_single_x(X[:, i], tag, event, discrete=func, **kws)
                                      }
            else:
                for i in range(X.shape[-1]):
                    func, kws = discrete[i]
                    result[label[i]] = {'woe': self.woe_single_x(X[:, i], tag, event, discrete=func, **kws),
                                        'iv': self.iv_single_x(X[:, i], tag, event, discrete=func, **kws)
                                        }
        else:
            if not label:
                for i in range(X.shape[-1]):
                    result[str(i)] = {'woe': self.woe_single_x(X[:, i], tag, event),
                                      'iv': self.iv_single_x(X[:, i], tag, event)
                                      }
            else:
                for i in range(X.shape[-1]):
                    result[label[i]] = {'woe': self.woe_single_x(X[:, i], tag, event),
                                        'iv': self.iv_single_x(X[:, i], tag, event)
                                        }

        result = OrderedDict(sorted(result.items(), key=lambda t: t[1]["iv"], reverse=True))
        return result

    def __init__(self, min_v=-20, max_v=20):
        self.WOE_MIN = min_v
        self.WOE_MAX = max_v

    def woe_single_x(self, x: np.ndarray, tag: np.ndarray, event: Any=1, discrete: Callable = None, **kwargs)->Dict[Any, float]:
        '''
        对单独一项自变量(列,特征)计算其woe和iv值.
        woe计算公式:

        :math: `woe_i = log(\frac {\frac {Bad_i} {Bad_{total}}} {\frac {Good_i} {Good_{total}}})`
        '''
        woe_dict = Woe.weight_of_evidence(x, tag, event,
                                          woe_min=self.WOE_MIN, woe_max=self.WOE_MAX, discrete=discrete, **kwargs)

        return woe_dict

    def iv_single_x(self, x: np.ndarray, tag: np.ndarray, event: Any=1,
                    discrete: Callable = None, **kwargs)->float:
        iv = Woe.information_value(x, tag, event, woe_min=self.WOE_MIN,
                                   woe_max=self.WOE_MAX, discrete=discrete, **kwargs)
        return iv

    def woe(self, X: np.ndarray, tag: np.ndarray, event: Any=1, label=None,
            discrete: Union[None, Tuple[Callable, Dict], List[Tuple[Callable, Dict]]] = None)->Dict[Any, Dict[Any, float]]:
        if label:
            len(label) != X.shape[-1]
            raise AttributeError("label must have the same len with the features' number")

        result = {}
        if discrete:
            if isinstance(discrete, tuple):
                discrete = [discrete for i in range(X.shape[-1])]
            if not label:
                for i in range(X.shape[-1]):
                    print(discrete[i])
                    func, kws = discrete[i]
                    result[str(i)] = self.woe_single_x(X[:, i], tag, event, discrete=func, **kws)
            else:
                for i in range(X.shape[-1]):
                    print(discrete[i])
                    func, kws = discrete[i]
                    result[label[i]] = self.woe_single_x(X[:, i], tag, event, discrete=func, **kws)

        else:
            if not label:
                for i in range(X.shape[-1]):
                    result[str(i)] = self.woe_single_x(X[:, i], tag, event)
            else:
                for i in range(X.shape[-1]):
                    result[label[i]] = self.woe_single_x(X[:, i], tag, event)
        return result

    def iv(self, X: np.ndarray, tag: np.ndarray, event: Any=1, label=None,
           discrete: Union[None, Tuple[Callable, Dict], List[Tuple[Callable, Dict]]] = None)->Dict[Any, float]:
        if label:
            len(label) != X.shape[-1]
            raise AttributeError("label must have the same len with the features' number")
        result = {}
        if discrete:
            if isinstance(discrete, tuple):
                discrete = [discrete for i in range(X.shape[-1])]
            if not label:
                for i in range(X.shape[-1]):
                    print(discrete[i])
                    func, kws = discrete[i]
                    result[str(i)] = self.iv_single_x(X[:, i], tag, event, discrete=func, **kws)
            else:
                for i in range(X.shape[-1]):
                    print(discrete[i])
                    func, kws = discrete[i]
                    result[label[i]] = self.iv_single_x(discrete(X[:, i]), tag, event, discrete=func, **kws)
        else:
            if not label:
                for i in range(X.shape[-1]):
                    result[str(i)] = self.iv_single_x(X[:, i], tag, event)
            else:
                for i in range(X.shape[-1]):
                    result[label[i]] = self.iv_single_x(X[:, i], tag, event)
        result = OrderedDict(sorted(result.items(), key=lambda t: t[1], reverse=True))
        return result
