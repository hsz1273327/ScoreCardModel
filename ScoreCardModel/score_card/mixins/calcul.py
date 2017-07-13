#coding:utf-8
class CalculScoreMixin(object):
    """需要有self._model,self.woe,self.tag"""
    def calcul_score(self,x,b=100,o=1,p=20):
        """计算已经用woe值替代好的无标签数据的得分"""
        factor= p/np.log(2)
        offset=b-p*(np.log(o)/np.log(2))
        p_f,p_t = self._model.predict_proba([x])[0]
        odds = p_t/p_f
        return factor*np.log(odds)+offset

    def get_score(self,orgx,b=100,o=1,p=20):
        """计算原始分好woe类特征数据的得分"""
        x = []
        if isinstance(orgx,dict):
            for k,v in orgx.items():
                value = self.woe.get(k).get(v)
                if value:
                    x.append(value)
                else:
                    x.append(0)
        else:
            for k,v in enumerate(orgx):
                value = self.woe.get(str(k)).get(v)
                if value:
                    x.append(value)
                else:
                    x.append(0)
        x = np.array(x)
        return self.calcul_score(x,b=b,o=o,p=p)

    def get_scores(self,b=100,o=1,p=20):
        """计算训练参数X的各条数据得分"""
        scores_ = []
        for i in range(self.woe_X.shape[0]):
            score = self.calcul_score(self.woe_X[i],b=b,o=o,p=p)
            class_ = self.tag[i]
            scores_.append((score,class_))
        return scores_

    def get_scores_x(self,b=100,o=1,p=20):
        """计算训练参数X的各条数据得分,并返回x的下标"""
        scores_ = []
        for i in range(self.woe_X.shape[0]):
            score = self.calcul_score(self.woe_X[i],b=b,o=o,p=p)
            class_ = self.tag[i]
            scores_.append((score,class_,i))
        return scores_
    def get_threshold(self,orgx,where = None,b=100,o=1,p=20):
        """计算原始分好woe类阈值点位置,参数为阈值点百分比位置的列表"""
        scores = sorted(self.get_score(orgx,b=b,o=o,p=p),reverse=True)
        if where:
            return tuple([scores[:int((i/100)*len(scores))][-1] for i in where])
        else:
            return tuple([scores[:int((i/100)*len(scores))][-1] for i in [3,5,7]])


class CalculProbabilityMixin(object):
    """需要有self._model,self.woe,self.tag"""
    def get_pt(self,orgx ,model="LogisticRegression"):
        """计算原始分好woe类特征数据的得分"""
        x = []
        if isinstance(orgx,dict):
            for k,v in orgx.items():
                value = self.woe.get(k).get(v)
                if value:
                    x.append(value)
                else:
                    x.append(0)
        else:
            for k,v in enumerate(orgx):
                value = self.woe.get(str(k)).get(v)
                if value:
                    x.append(value)
                else:
                    x.append(0)
        x = np.array(x)
        _,p_t = self.models.get(model).predict_proba([x])[0]
        return p_t

    def get_threshold_pt(self,orgX,where = None,model="LogisticRegression"):
        """计算原始分好woe类特征数据阈值点位置,参数为阈值点百分比位置的列表"""
        scores_ = [self.get_pt(i,model=model) for i in orgX]
        scores = sorted(scores_,reverse=True)
        if where:
            return tuple([scores[:int((i/100)*len(scores))][-1] for i in where])
        else:
            return tuple([scores[:int((i/100)*len(scores))][-1] for i in [3,5,7]])

    def get_threshold_pt_trained(self,where = None,model="LogisticRegression"):
        """计算训练数据阈值点位置,参数为阈值点百分比位置的列表"""
        scores_ = []
        for i in range(self.woe_X.shape[0]):
            p_f,p_t = self.models.get(model).predict_proba([self.woe_X[i]])[0]

            scores_.append(p_t)

        scores = sorted(scores_,reverse=True)
        if where:
            return tuple([scores[:int((i/100)*len(scores))][-1] for i in where])
        else:
            return tuple([scores[:int((i/100)*len(scores))][-1] for i in [3,5,7]])
