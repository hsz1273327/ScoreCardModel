#coding:utf-8
import pickle

class ToDictMixin(object):
    def to_dict(self):
        init = dict(
            event = self.event,
            discrete = self.discrete,
            feature_order = self.feature_order,
            label = self.label,
            min_v = self.min_v,
            max_v =self.max_v
        )
        X = self.X.tolist(),
        tag = self.tag.tolist(),
        model =  pickle.dumps(self._model,protocol=2)

        result = {
            'init':init,
            'model':model,
            'X':X,
            'tag':tag,
            "model_type":self._model_type.__name__
        }
        return result
class SerializeMixin(object):
    """需要to_dict"""
    def dumps(self):
        result = pickle.dumps(self.to_dict(),protocol=2)
        return result

    def dump(self,f):
        pickle.dump(self.to_dict(),f,protocol=2)
