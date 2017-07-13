#coding:utf-8
import pickle
import numpy as np
class ConstructorMixin(object):

    @classmethod
    def from_dict(cls,js_dict):

        model_type_name = js_dict.get("model_type")
        X=np.array(js_dict.get("X")[0])
        tag = np.array(js_dict.get("tag")[0])
        if model_type_name == "SVC":
            from sklearn.svm import SVC
            model_type = SVC
        elif model_type_name == "LogisticRegression":
            from sklearn.linear_model import LogisticRegression
            model_type = LogisticRegression
        else:
            from sklearn.linear_model import LogisticRegression
            model_type = LogisticRegression

        obj = cls(X=X,tag=tag,model_type=model_type,**js_dict.get("init"))
        obj._model =pickle.loads(js_dict.get("model"))
        return obj

    @classmethod
    def from_pickle_loads(cls,pickle_str):
        pickle_dict=pickle.loads(pickle_str)
        obj = cls.from_dict(pickle_dict)
        return obj

    @classmethod
    def from_pickle_load(cls,f):
        pickle_dict=pickle.load(f)
        obj = cls.from_dict(pickle_dict)
        return obj
