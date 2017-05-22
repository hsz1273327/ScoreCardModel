__all__ = ["Woe"]
import numpy as np
from collections import OrderedDict
from typing import Any, Union, Sequence, Tuple, Dict, Callable, List
from score_card_model.utils.discretization.sharing import discrete
from score_card_model.utils.check import check_array_binary, check_array_continuous
from score_card_model.utils.count import count_binary


class Woe:
    """
    注意woe和iv方法如果有discrete参数,必须是tuple(func,dict)或者list(tuple(func,dict)).

    使用的时候先实例化,然后使用`iv`或者`woe`或者调用`__call__`来获取其对应的值.

    Property:
        WOE_MIN (number): - WOE的最小值,如果rate_event为0就将它替换为woe_min
        WOE_MAX (number): - WOE的最大值,如果ratenon_event为0就将它替换为woe_max
        X (np.ndarray): - 输入的2-D矩阵
        tag (np.ndarray): - 输入的1-D数组,布尔标签数据
        event (Any): - 为True的标签
        label (Union[None, List[str]]): - 各项特征的名字列表
        discrete (Union[None, Tuple[Callable, Dict], List[Tuple[Callable, Dict]]]): - 处理各项特征分区间的tuple(func,args)数据
    """

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
        对单独一项自变量(列,特征)计算其woe值.
        woe计算公式:
        $$ woe_i = log(\frac {\frac {Bad_i} {Bad_{total}}} {\frac {Good_i} {Good_{total}}}) $$
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
        '''
        对单独一项自变量(列,特征)计算其woe和iv值.
        iv计算公式:
        $$ IV_i=({\frac {Bad_i}{Bad_{total}}}-{\frac{Good_i}{Good_{total}}})*log(\frac{\frac{Bad_i}{Bad_{total}}}{\frac{Good_i}{Good_{total}}}) $$

        $$ IV = \sum_{k=0}^n IV_i $$
        '''

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

    @property
    def WOE_MIN(self):
        return self.__WOE_MIN
    @property
    def WOE_MAX(self):
        return self.__WOE_MAX
    @property
    def X(self):
        return self.__X
    @property
    def tag(self):
        return self.__tag
    @property
    def event(self):
        return self.__event
    @property
    def label(self):
        return self.__label
    @property
    def discrete(self):
        return self.__discrete




    def __init__(self,X: np.ndarray, tag: np.ndarray, event: Any=1, label:Union[None, List[str]]=None,
                 discrete: Union[None, Tuple[Callable, Dict], List[Tuple[Callable, Dict]]] = None,
                 min_v=-20, max_v=20):
        """
        初始化一个woe对象.

        Parameters:
            min_v (number): - WOE的最小值,如果rate_event为0就将它替换为woe_min
            max_v (number): - WOE的最大值,如果ratenon_event为0就将它替换为woe_max
            X (np.ndarray): - 输入的2-D矩阵
            tag (np.ndarray): - 输入的1-D数组,布尔标签数据
            event (Any): - 为True的标签
            label (Union[None, List[str]]): - 各项特征的名字列表
            discrete (Union[None, Tuple[Callable, Dict], List[Tuple[Callable, Dict]]]): - 处理各项特征分区间的tuple(func,args)数据

        Raises:
            AttributeError("label must have the same len with the features' number") : - label如果长度与特征数不符则报错

            AttributeError("discrete method list must have the same len with the features' number") : - discrete如果是列表,如果其长度与特征数不符则报错

        """
        if label:
            len(label) != X.shape[-1]
            raise AttributeError("label must have the same len with the features' number")

        if isinstance(discrete,List):
            if len(discrete) != X.shape[-1]:
                raise AttributeError("discrete method list must have the same len with the features' number")
        self.__WOE_MIN = min_v
        self.__WOE_MAX = max_v
        self.__X = X
        self.__tag = tag
        self.__event = event
        self.__label = label
        self.__discrete = discrete
        self.__woe = None
        self.__iv  = None



    def __calcul(self,func):
        """用于计算iv或者woe
        """
        result = {}
        discrete = self.discrete
        if discrete:
            if isinstance(discrete, tuple):
                discrete = [discrete for i in range(self.X.shape[-1])]
            if not self.label:
                for i in range(self.X.shape[-1]):

                    if len(discrete[i]) == 1:
                        discretefunc = discrete[i][0]
                        result[str(i)] = func(self.X[:, i], self.tag, self.event, woe_min=self.WOE_MIN, woe_max=self.WOE_MAX,discrete=discretefunc)
                    elif len(discrete[i]) == 2:
                        discretefunc, kws = discrete[i]
                        result[str(i)] = func(self.X[:, i], self.tag, self.event, woe_min=self.WOE_MIN, woe_max=self.WOE_MAX,discrete=discretefunc, **kws)
                    else:
                        raise AttributeError("discrete argument must a tuple of the objects :func,args")
            else:
                for i in range(self.X.shape[-1]):

                    if len(discrete[i]) == 1:
                        discretefunc = discrete[i]
                        result[label[i]] = func(self.X[:, i], self.tag, self.event, woe_min=self.WOE_MIN, woe_max=self.WOE_MAX,discrete=discretefunc)
                    elif len(discrete[i]) == 2:
                        discretefunc, kws = discrete[i]
                        result[label[i]] = func(self.X[:, i], self.tag, self.event, woe_min=self.WOE_MIN, woe_max=self.WOE_MAX,discrete=discretefunc, **kws)
                    else:
                        raise AttributeError("discrete argument must a tuple of the objects :func[,args]")
        else:
            if not self.label:
                for i in range(self.X.shape[-1]):
                    result[str(i)] = func(self.X[:, i], self.tag, self.event,woe_min=self.WOE_MIN, woe_max=self.WOE_MAX)
            else:
                for i in range(self.X.shape[-1]):
                    result[label[i]] = func(self.X[:, i], self.tag, self.event,woe_min=self.WOE_MIN, woe_max=self.WOE_MAX)
        return result
    @property
    def iv(self)->Dict[Any, float]:

        if self.__iv :
            return self.__iv
        result = self.__calcul(Woe.information_value)
        result = OrderedDict(sorted(result.items(), key=lambda t: t[1], reverse=True))

        self.__iv = result
        return result

    @property
    def woe(self)->Dict[Any, Dict[Any, float]]:
        if self.__woe:
            return self.__woe
        result = self.__calcul(Woe.weight_of_evidence)
        self.__woe = result
        return result

    def __call__(self)->Dict[Any, Dict[Any, float]]:

        iv = self.iv
        woe = self.woe
        result = {}
        for l,_ in iv.items():
            result[l] = {"iv":iv.get(l),
                      "woe":woe.get(l)
                      }
        result = OrderedDict(sorted(result.items(), key=lambda t: t[1]["iv"], reverse=True))
        return result
