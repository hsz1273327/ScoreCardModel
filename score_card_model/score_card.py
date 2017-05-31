from typing import Any, Union, Sequence, Tuple, Dict, Callable, List
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from score_card_model.feature_selection.weight_of_evidence import Woe
import matplotlib.pyplot as plt
class ScoreCard:
    def __init__(self,X: np.ndarray, tag: np.ndarray, event: Any=1,
                 label:Union[None, List[str]]=None,
                 discrete: Union[None, Tuple[Callable, Dict], List[Tuple[Callable, Dict]]] = None,
                 min_v=-20, max_v=20):
        self.models = {}
        self.N = X.shape[-1]
        self.X = X
        self.tag = tag
        woe = Woe(X=X, tag=tag, event=event,label=label,discrete=discrete,min_v=min_v, max_v=max_v)
        self.woe = woe.woe
        self.X_label = woe.X_result
        woe_X = []
        if label:
            for i in range(self.X.shape[-1]):
                array = self.X_label[:, i]
                woe_i = woe.get(label[i])
                line = np.array([woe_i.get(i) for i in array])
                woe_X.append(line)

        else:
            for i in range(self.X.shape[-1]):
                array = self.X_label[:, i]
                woe_i = self.woe.get(str(i))
                line = np.array([woe_i.get(i) for i in array])
                woe_X.append(line)

        self.woe_X = np.array(woe_X).T

    def test_fit(self,Model = LogisticRegression,times = 10,test_size=0.4, random_state=0,**kwargs):
        scores = []
        for i in range(times):
            X_train,X_test,y_train,y_test = cross_validation.train_test_split(self.woe_X,
                                                                              self.tag,
                                                                              test_size=test_size,
                                                                              random_state=random_state)
            model = Model(**kwargs)
            model.fit (X_train,y_train)
            score = model.score(X_test,y_test)
            scores.append(score)
        return np.mean(scores)
    def fit(self,Model = LogisticRegression,name=None,**kwargs):
        model = Model(**kwargs)
        model.fit(self.woe_X,self.tag)
        if name:
            self.models[name] = model
        else:

            self.models[Model.__name__] = model
        return model

    def get_score(self,orgx ,model="LogisticRegression",b=100,o=1,p=20):
        """计算原始分好类特征数据的得分"""
        x = []
        if isinstance(orgx,dict):
            for k,v in orgx.items():
                value = self.woe.get(k).get(v)

            x.append(value)
        else:
            for k,v in enumerate(orgx):
                value = self.woe.get(str(k)).get(v)

            x.append(value)
        x = np.array(x)
        return self.calcul_score(x ,model=model,b=b,o=o,p=p)

    def calcul_score(self,x ,model="LogisticRegression",b=100,o=1,p=20):
        """计算已经用woe值替代好的无标签数据的得分"""
        factor= p/np.log(2)
        offset=b-p*(np.log(o)/np.log(2))
        p_f,p_t = self.models.get(model).predict_proba(x)[0]
        odds = p_t/p_f
        return factor*np.log(odds)+offset

    def get_scores(self,model="LogisticRegression",b=100,o=1,p=20):
        """计算参数X的各条数据得分"""
        scores_ = []
        for i in range(self.woe_X.shape[0]):
            score = self.calcul_score(self.woe_X[i],model=model,b=b,o=o,p=p)
            class_ = self.tag[i]
            scores_.append((score,class_))
        return scores_

    def get_model_info(model = "LogisticRegression"):
        coef = self.model[model].coef_
        params = self.model[model].get_params()
        return {
            'coef':coef,
            'params': params
        }

    def get_ks(self,model="LogisticRegression",b=100,o=1,p=20,n=10):

        scores = sorted(self.get_scores(model=model,b=b,o=o,p=p),reverse=True)
        bad_total = sum([1 for score,class_ in scores if class_ == 0])
        good_total = len(scores)-bad_total
        result = []
        for i in range(n):
            limit = int((i/n)*len(scores))
            temp = scores[:limit]
            bad =  sum([1 for score,class_ in temp if class_ == 0])
            good = len(temp)-bad
            bad_rat = bad/bad_total
            good_rat= good/good_total
            ks = abs(good_rat-bad_rat)
            result.append((round(limit/len(scores),3),good_rat,bad_rat,ks))
        return result

    def drawks(self,model="LogisticRegression",b=100,o=1,p=20,n=10):
        kss = self.get_ks(model=model,b=b,o=o,p=p,n=n)
        X = [i for i,_,_,_ in kss]
        Y_good = [i for _,i,_,_ in kss]
        Y_bad = [i for _,_,i,_ in kss]
        Y_ks = [i for _,_,_,i in kss]
        plt.plot(X,Y_good,color="blue")
        plt.plot(X,Y_bad,color="red")
        plt.plot(X,Y_ks,color="yellow")
        plt.legend(loc='upper left')
        plt.show()
