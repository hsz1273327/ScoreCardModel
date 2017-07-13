#coding:utf-8
import math
class KSMixin(object):
    """需要self.get_scores_x,self.get_scores"""
    def get_bad(self,rat=0.05,b=100,o=1,p=20,n=100):
        scores = sorted(self.get_scores_x(b=b,o=o,p=p),reverse=True)
        l = len(scores)
        limit=math.ceil(rat*l)
        return [i for i in scores[:limit] if i[1]== 0]

    def get_ks(self,b=100,o=1,p=20,n=10):

        scores = sorted(self.get_scores(b=b,o=o,p=p),reverse=True)
        bad_total = sum([1 for score,class_ in scores if class_ == 0])
        good_total = len(scores)-bad_total
        result = []
        for i in range(n):
            limit = int((i/n)*len(scores))
            temp = scores[:limit]
            roc=0
            bad =  sum([1 for score,class_ in temp if class_ == 0])
            good = len(temp)-bad
            bad_rat = bad/bad_total
            good_rat= good/good_total
            ks = abs(good_rat-bad_rat)
            if (good+bad) == 0:
                good_bad_rate = 0
            else:
                good_bad_rate = good/(good+bad)
            result.append((round(limit/len(scores),3),good_rat,bad_rat,ks,good_bad_rate,roc))
        return result
